import os
import yaml
import shutil
import corner
import numpy as np
import pandas as pd
import autofit as af
import anesthetic as an
import bayesnova.old_src.preprocessing as prep

from pyprojroot import here
from bayesnova.analysis import Analysis
from bayesnova.base import UnivariateGaussian
from bayesnova.cosmo import FlatLambdaCDM, FlatwCDM
from bayesnova.progenitors import SNeProgenitors
from bayesnova.calibration import Tripp, TrippDust, OldTripp
from bayesnova.mixture import ConstantWeighting, LogisticLinearWeighting, RedshiftDependentWeighting, TwoPopulationMixture

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"

workspace_path = str(here())
os.chdir(workspace_path)

def main():

    dataset = 'pantheon_full_fiducial_volume_limited'
    name = "independent_progenitor"

    try:
        shutil.rmtree("/groups/dark/osman/BayeSNova/output/" + dataset + "/" + name)
    except:
        pass

    with open("/home/jacob/Uni/Msc/Thesis/BayeSNova/examples/configs/config.yaml") as f:
        old_config = yaml.safe_load(f)

    # Load data
    #data_path = "/groups/dark/osman/Thesis_old/data/processed_data/supercal"
    data_path = f"/groups/dark/osman/Msc_Thesis/src/data/{dataset}/{dataset}.dat"
    data = pd.read_csv(data_path, sep=" ")

    volumetric_rates_path = "/groups/dark/osman/Msc_Thesis/src/data/volumetric_rates.dat"
    volumetric_rates_df = pd.read_csv(
        volumetric_rates_path, sep=" "
    )

    old_config['model_cfg']['use_volumetric_rates'] = True
    old_config['model_cfg']['host_galaxy_cfg']['use_properties'] = True
    old_config['model_cfg']['host_galaxy_cfg']['independent_property_names'] = ['global_mass','t',]

    prep.init_global_data(data, volumetric_rates_df, old_config['model_cfg'])
    
    print(f"N SNe: {len(prep.sn_observables)}")
    not_calibrator_indeces = ~prep.idx_calibrator_sn
    print(f"N Calibrators: {np.sum(~not_calibrator_indeces)}")
    sn_observables = prep.sn_observables[not_calibrator_indeces]
    sn_redshifts = prep.sn_redshifts[not_calibrator_indeces]
    sn_covariances = prep.sn_covariances[not_calibrator_indeces]
    print(f"N SNe after removing calibrators: {len(sn_observables)}")

    volumetric_rate_redshifts = prep.observed_volumetric_rate_redshifts
    volumetric_rate_observations = prep.observed_volumetric_rates
    volumetric_rate_errors = prep.observed_volumetric_rate_errors

    apparent_B_mag = sn_observables[:,0]
    stretch = sn_observables[:,1]
    color = sn_observables[:,2]
    redshift = sn_redshifts
    observed_covariance = sn_covariances
    host_properties = prep.host_galaxy_observables[not_calibrator_indeces]
    host_covariances = prep.host_galaxy_covariances[not_calibrator_indeces]
    
    # SNe Ia model

    cosmology = af.Model(
        FlatwCDM,
        H0=73.0,
        #Om0=0.3,
        Tcmb0=2.725,
        Neff=3.046,
        m_nu_1=0.0,
        m_nu_2=0.0,
        m_nu_3=0.06,
        Ob0=0.04897,
        w0=-1.0
    )

    sn_pop_1 = af.Model(
        TrippDust,
        cosmology=cosmology,
        peculiar_velocity_dispersion=200.0,
        sigma_M_int=0.,
        R_B_min=1.5,
    )

    sn_pop_2 = af.Model(
        TrippDust,
        cosmology=cosmology,
        peculiar_velocity_dispersion=200.0,
        sigma_M_int=0.,
        R_B_min=1.5,
    )

    progenitor_model_weight = af.Model(
        SNeProgenitors,
        cosmology=cosmology,
        #eta=1.02e-4
    )

    progenitor_model_rate = af.Model(
        SNeProgenitors,
        cosmology=cosmology,
        #eta=1.02e-4
    )

    sn_pop_2.cosmology.Om0 = sn_pop_1.cosmology.Om0
    sn_pop_2.cosmology.w0 = sn_pop_1.cosmology.w0
    progenitor_model_weight.cosmology.Om0 = sn_pop_1.cosmology.Om0
    progenitor_model_weight.cosmology.w0 = sn_pop_1.cosmology.w0
    progenitor_model_rate.cosmology.Om0 = sn_pop_1.cosmology.Om0
    progenitor_model_rate.cosmology.w0 = sn_pop_1.cosmology.w0

    sn_weighting_model = af.Model(
        RedshiftDependentWeighting,
        progenitor_model=progenitor_model_weight,
    )
    # sn_weighting_model.progenitor_model.eta = progenitor_model.eta
    # sn_weighting_model.progenitor_model.f_prompt = progenitor_model.f_prompt

    # sn_weighting_model = af.Model(
    #     ConstantWeighting,
    # )

    sn_model = af.Model(
        TwoPopulationMixture,
        population_models=[sn_pop_1, sn_pop_2],
        weighting_model=sn_weighting_model
    )

    population_models = sn_model.population_models
    n_populations = len(population_models)
    reference_population_attributes = vars(population_models[0])
    shared_parameter_names = [
        "sigma_M_int",
        "alpha",
        "beta",
        "R_B",
        "sigma_R_B",
        "gamma_E_BV"
    ]
    
    for i in range(1,n_populations):
        population = population_models[i]
        population_attributes = vars(population)
        population_attributes['cosmology'] = reference_population_attributes['cosmology']
        
        for param in shared_parameter_names:
            population_attributes[param] = reference_population_attributes[param]

        population.__dict__.update(population_attributes)   

    sn_model.add_assertion(
        population_models[-1].stretch_int > population_models[0].stretch_int
    )

    sn_and_progenitor_model = af.Collection(
        sn=sn_model,
        progenitor_model=progenitor_model_rate
    )

    # Host mass model

    # host_mass_pop_1 = af.Model(
    #     UnivariateGaussian,
    # )

    # host_mass_pop_2 = af.Model(
    #     UnivariateGaussian,
    # )

    # host_mass_weighting_model = af.Model(
    #     LogisticLinearWeighting,
    #     #scale=0.
    #     offset=0.
    # )

    # host_mass_model = af.Model(
    #     TwoPopulationMixture,
    #     population_models=[host_mass_pop_1, host_mass_pop_2],
    #     weighting_model=host_mass_weighting_model
    # )

    # host_mass_model.population_models[0].mu = af.UniformPrior(lower_limit=6.0, upper_limit=16.0)
    # host_mass_model.population_models[1].mu = af.UniformPrior(lower_limit=6.0, upper_limit=16.0)

    # host_mass_model.add_assertion(
    #     host_mass_model.population_models[0].mu > host_mass_model.population_models[1].mu
    # )

    # host_morphology_pop_1 = af.Model(
    #     UnivariateGaussian,
    # )

    # host_morphology_pop_2 = af.Model(
    #     UnivariateGaussian,
    # )

    # host_morphology_weighting_model = af.Model(
    #     LogisticLinearWeighting,
    #     offset=0.
    # )

    # host_morphology_model = af.Model(
    #     TwoPopulationMixture,
    #     population_models=[host_morphology_pop_1, host_morphology_pop_2],
    #     weighting_model=host_morphology_weighting_model
    # )

    # host_morphology_model.population_models[0].mu = af.UniformPrior(lower_limit=-10., upper_limit=10.)
    # host_morphology_model.population_models[1].mu = af.UniformPrior(lower_limit=-10., upper_limit=10.)

    # host_morphology_model.add_assertion(
    #     host_morphology_model.population_models[0].mu < host_morphology_model.population_models[1].mu
    # )

    # sn_and_host_model = af.Collection(
    #     sn=sn_model,
    #     host_models=[host_mass_model],# host_morphology_model]
    # )

    model = sn_and_progenitor_model #sn_model #sn_and_host_model #sn_and_progenitor_model

    analysis = Analysis(
        apparent_B_mag=apparent_B_mag,
        stretch=stretch,
        color=color,
        redshift=redshift,
        observed_covariance=observed_covariance,
        host_properties=host_properties,
        host_covariances=host_covariances,
        volumetric_rate_redshifts=volumetric_rate_redshifts,
        volumetric_rate_observations=volumetric_rate_observations,
        volumetric_rate_errors=volumetric_rate_errors,
        use_log_marginalization=True,
        use_truncated_R_B_prior=True,
        gamma_quantiles_cfg={
            "lower": 1.,
            "upper": 20.,
            "resolution": 0.001,
            "cdf_limit": 0.995
        },
    )

    instance_created = False
    while not instance_created:
        try:
            instance = model.random_instance_from_priors_within_limits()
            llh_value = analysis.log_likelihood_function(instance)
            if llh_value == -1e99:
                pass
            else:
                instance_created = True
        except:
            pass
    
    # analysis.log_likelihood_function(instance)
    print("\nModel Info:")
    print(model.info)
    print("\n")

    search = af.Nautilus(
        path_prefix=dataset,
        name=name,
        n_live=512,#int(2*4096), #768,#1024,#4096,
        n_batch=512,#384,#512,#1024,
        mpi=True,
        n_eff=5000,#int(1e4),
        number_of_cores=512,#256,#512,
        discard_exploration=True,
        iterations_per_update=int(1e10),
        split_threshold=10.,
    )

    # search = af.DynestyStatic(
    #     path_prefix=dataset,
    #     name=name,
    #     nlive=496,#5000,
    #     sample='rwalk',
    #     number_of_cores=496,#257,
    #     iterations_per_update=int(1e10),
    #     walks=50,
    #     slices=50,
    # )

    result = search.fit(model=model, analysis=analysis)

    print("\nResult:")
    print(result.info)

    sampler = result.samples
    samples = np.array(sampler.parameter_lists)
    weights = np.array(sampler.weight_list)
    llh = np.array(sampler.log_likelihood_list)
    n_live = 1024#4096
    param_names = model.model_component_and_parameter_names
    anesthetic_samples = an.NestedSamples(
        data=samples, weights=weights, logL=llh, logL_birth=n_live, columns=param_names
    )

    bayesian_stats = anesthetic_samples.stats(1000)
    stat_names = [tup[0] for tup in list(bayesian_stats.keys())]
    percs = np.percentile(bayesian_stats.values, [16, 50, 84], axis=0)
    percs_df = pd.DataFrame(
        data=percs,
        index=['16th', '50th', '84th'],
        columns=stat_names
    )

    save_path = str(search.paths)
    bayesian_stats.to_csv(save_path + "/bayesian_stats.csv")
    percs_df.to_csv(save_path + "/percentiles.csv")

    print("\nBayesian Stats:")
    for i in range(len(stat_names)):
        print(f"{stat_names[i]}: {percs[1,i]:.3f} +/- {0.5*(percs[2,i]-percs[0,i]):.3f}")


    
if __name__ == "__main__":
    main()