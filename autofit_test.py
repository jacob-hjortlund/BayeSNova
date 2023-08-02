import os
import yaml
import shutil
import corner
import numpy as np
import pandas as pd
import autofit as af
import bayesnova.preprocessing as prep

from pyprojroot import here
from src.analysis import Analysis
from base import UnivariateGaussian
from cosmo import FlatLambdaCDM
from calibration import Tripp, TrippDust, OldTripp
from mixture import ConstantWeighting, LogisticLinearWeighting, TwoPopulationMixture

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

workspace_path = str(here())
os.chdir(workspace_path)

def main():

    name = "test_rwalk_50_run_1_sn_host"
    try:
        shutil.rmtree("/groups/dark/osman/BayeSNova/output/" + name)
    except:
        pass

    with open("/groups/dark/osman/BayeSNova/examples/configs/config.yaml") as f:
        old_config = yaml.safe_load(f)

    # Load data
    data_path = "/groups/dark/osman/Thesis_old/data/processed_data/supercal"
    # data_path = "/groups/dark/osman/Msc_Thesis/src/data/supercal_hubble_flow/supercal_hubble_flow.dat"
    data = pd.read_csv(data_path, sep=" ")

    old_config['model_cfg']['host_galaxy_cfg']['use_properties'] = True
    old_config['model_cfg']['host_galaxy_cfg']['independent_property_names'] = ['global_mass']

    prep.init_global_data(data, None, old_config['model_cfg'])
    
    print(f"N SNe: {len(prep.sn_observables)}")
    not_calibrator_indeces = ~prep.idx_calibrator_sn
    print(f"N Calibrators: {np.sum(~not_calibrator_indeces)}")
    sn_observables = prep.sn_observables[not_calibrator_indeces]
    sn_redshifts = prep.sn_redshifts[not_calibrator_indeces]
    sn_covariances = prep.sn_covariances[not_calibrator_indeces]
    print(f"N SNe after removing calibrators: {len(sn_observables)}")

    apparent_B_mag = sn_observables[:,0]
    stretch = sn_observables[:,1]
    color = sn_observables[:,2]
    redshift = sn_redshifts
    observed_covariance = sn_covariances
    host_properties = prep.host_galaxy_observables[not_calibrator_indeces]
    host_covariances = prep.host_galaxy_covariances[not_calibrator_indeces]
    
    # SNe Ia model

    cosmology = af.Model(
        FlatLambdaCDM,
        H0=73.0,
        Om0=0.3
    )

    sn_pop_1 = af.Model(
        TrippDust,
        cosmology=cosmology,
        peculiar_velocity_dispersion=200.0,
        sigma_M_int=0.,
        # alpha=-0.206,
        # beta=2.936,
        # R_B=3.661,
        # sigma_R_B=0.685,
        # gamma_E_BV=4.385,
        # M_int=-19.462,
        # color_int=-0.064,
        # sigma_color_int=0.086,
        # stretch_int=-1.491,
        # sigma_stretch_int=0.646,
        # tau_E_BV=0.022,
    )

    sn_pop_2 = af.Model(
        TrippDust,
        cosmology=cosmology,
        peculiar_velocity_dispersion=200.0,
        sigma_M_int=0.,
        # alpha=-0.206,
        # beta=2.936,
        # R_B=3.661,
        # sigma_R_B=0.685,
        # gamma_E_BV=4.385,
        # M_int=-19.317,
        # color_int=-0.151,
        # sigma_color_int=0.031,
        # stretch_int=0.326,
        # sigma_stretch_int=0.647,
        # tau_E_BV=0.033,
    )

    sn_weighting_model = af.Model(
        ConstantWeighting
    )

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

    # Host mass model

    host_mass_pop_1 = af.Model(
        UnivariateGaussian,
    )

    host_mass_pop_2 = af.Model(
        UnivariateGaussian,
    )

    host_mass_weighting_model = af.Model(
        LogisticLinearWeighting,
    )

    host_mass_model = af.Model(
        TwoPopulationMixture,
        population_models=[host_mass_pop_1, host_mass_pop_2],
        weighting_model=host_mass_weighting_model
    )

    host_mass_model.population_models[0].mu = af.UniformPrior(lower_limit=6.0, upper_limit=16.0)
    host_mass_model.population_models[1].mu = af.UniformPrior(lower_limit=6.0, upper_limit=16.0)

    host_mass_model.add_assertion(
        host_mass_model.population_models[0].mu > host_mass_model.population_models[1].mu
    )

    sn_and_host_model = af.Collection(
        sn=sn_model,
        host_models=[host_mass_model]
    )

    model = sn_and_host_model

    instance_created = False
    while not instance_created:
        try:
            instance = model.random_instance_from_priors_within_limits()
            instance_created = True
        except:
            pass

    print("\nModel Info:")
    print(model.info)
    print("\n")

    analysis = Analysis(
        apparent_B_mag=apparent_B_mag,
        stretch=stretch,
        color=color,
        redshift=redshift,
        observed_covariance=observed_covariance,
        host_properties=host_properties,
        host_covariances=host_covariances,
    )

    # instance = sn_model.instance_from_unit_vector([])
    instance_created = False
    while not instance_created:
        try:
            instance = model.random_instance_from_priors_within_limits()
            instance_created = True
        except:
            pass

    print("\nInstance:")
    param_names = model.parameter_names
    
    for param in param_names:
        for i, model in enumerate(instance.population_models):
            model_vars = vars(model)
            model_name = f"Pop {i+1}"
            if param in model_vars.keys():
                print(f"{model_name}: {param} = {model_vars[param]}")
        print("\n")

        weight_model_vars = vars(instance.weighting_model)
        if param in weight_model_vars.keys():
            print(f"Weighting: {param} = {weight_model_vars[param]}")
        
    print(analysis.log_likelihood_function(instance=instance))

    # search = af.Emcee(
    #     name=name,
    #     nwalkers=256,
    #     nsteps=100000,
    #     initializer=af.InitializerBall(lower_limit=0.4995, upper_limit=0.5005),
    #     number_of_cores=257,
    #     iterations_per_update=int(1e100),
    # )

    search = af.DynestyStatic(
        name=name,
        nlive=1000,
        sample='rwalk',
        number_of_cores=257,
        iterations_per_update=int(1e100),
        walks=50,
        slices=50
        #dlogz=500.
    )

    result = search.fit(model=model, analysis=analysis)

    print("\nResult:")
    print(result.info)

    
if __name__ == "__main__":
    main()