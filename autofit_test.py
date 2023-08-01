import os
import yaml
import shutil
import numpy as np
import pandas as pd
import autofit as af
import bayesnova.preprocessing as prep

from pyprojroot import here
from src.analysis import Analysis
from base_models import ConstantWeighting
from src.cosmo_models import FlatLambdaCDM
from src.sn_models import Tripp, TrippDust, OldTripp, TwoSNPopulation

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

workspace_path = str(here())
os.chdir(workspace_path)

def main():

    name = "test_rwalk_50_run_4"
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
    
    cosmology = af.Model(
        FlatLambdaCDM,
        H0=73.0,
        Om0=0.3
    )

    pop_1 = af.Model(
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

    pop_2 = af.Model(
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

    weighting_model = af.Model(
        ConstantWeighting,
        #weight=0.3178
    )

    sn_model = af.Model(
        TwoSNPopulation,
        population_models=[pop_1, pop_2],
        weighting_model=weighting_model
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

    print("\nSN model pre-fit:")
    print(sn_model.info)
    print("\n")

    analysis = Analysis(
        apparent_B_mag=apparent_B_mag,
        stretch=stretch,
        color=color,
        redshift=redshift,
        observed_covariance=observed_covariance,
    )

    # instance = sn_model.instance_from_unit_vector([])
    instance_created = False
    while not instance_created:
        try:
            instance = sn_model.random_instance_from_priors_within_limits()
            instance_created = True
        except:
            pass

    print("\nInstance:")
    param_names = sn_model.parameter_names
    
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

    result = search.fit(model=sn_model, analysis=analysis)

    # print("\nSN model:")
    # print(sn_model.info)
    # print("\n")

    print("\nResult:")
    print(result.info)

    # import time
    # from dynesty import plotting as dyplot

    # samples = result.samples

    # fig, _ = dyplot.cornerplot(
    #     results=samples.results_internal,
    #     labels=param_names
    # )
    # fig.savefig("corner.png")

    # N = 1000
    # instances = [
    #     samples.from_sample_index(sample_index=i) for i in range(len(samples))[:N]
    # ]
    # t0 = time.time()
    # for i in range(len(instances)):
    #     analysis.log_likelihood_function(instance=instances[i])
    # t1 = time.time()
    # print(f"Time per likelihood evaluation: {(t1-t0)/N}")
    
if __name__ == "__main__":
    main()