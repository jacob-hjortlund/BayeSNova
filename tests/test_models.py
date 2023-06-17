import pytest
import numpy as np
import src.model as models
import src.preprocessing as prep

NULL_VALUE = -9999.

@pytest.mark.parametrize(
    (
        (
            "monkeypatch, dict2class, " + 
            "observables_and_covariances, expected_mvg_gamma_convolution, " +
            "inputs_config, Rb, sig_Rb, tau_Ebv, gamma_Ebv"
        )
    ),
    [
        (
            "monkeypatch",
            "dict2class",
            "observables_and_covariances",
            "expected_mvg_gamma_convolution",
            ( (10, 3, 3), 10, False, True, False ),
            4.1, 3., 0.1, 3.
        ),
    ], indirect=[
        "monkeypatch",
        "dict2class",
        "observables_and_covariances",
        "expected_mvg_gamma_convolution"
    ]
)
def test_SNmodel_Ebv_convolution(
    monkeypatch, dict2class,
    observables_and_covariances, expected_mvg_gamma_convolution,
    inputs_config, Rb, sig_Rb, tau_Ebv, gamma_Ebv
):
    
    n = inputs_config[0][0]
    Ebv = NULL_VALUE
    tau_Rb_1 = tau_Rb_2 = NULL_VALUE
    gamma_Rb_1 = gamma_Rb_2 = NULL_VALUE
    shift_Rb = NULL_VALUE
    upper_bound = 10.

    monkeypatch.setattr(
        models.Model, "get_upper_bounds",
        lambda *args: (np.ones(n) * upper_bound, np.ones(n) * upper_bound)
    )
    cfg = {
        "Rb_integral_lower_bound": 0.,
        "Ebv_integral_lower_bound": 0.,
    }
    prep.global_model_cfg = dict2class(cfg)
    prep.gRb_quantiles = np.ones(n) * 0.5
    prep.gEbv_quantiles = np.ones(n) * 0.5
    prep.selection_bias_correction = np.ones(n)

    observables, covariances = observables_and_covariances(*inputs_config)
    expected_convolution_outputs = expected_mvg_gamma_convolution(
        observables, covariances, Rb, sig_Rb,
        tau_Ebv, gamma_Ebv, upper_bound
    )

    model = models.Model()
    model.convolution_fn = models._Ebv_prior_convolution

    convolution_outputs = model.prior_convolutions(
        covs_1=covariances, covs_2=covariances,
        residuals_1=observables, residuals_2=observables,
        Rb_1=Rb, Rb_2=Rb, sig_Rb_1=sig_Rb, sig_Rb_2=sig_Rb,
        tau_Rb_1=tau_Rb_1, tau_Rb_2=tau_Rb_2,
        gamma_Rb_1=gamma_Rb_1, gamma_Rb_2=gamma_Rb_2,
        shift_Rb=shift_Rb,
        Ebv_1=Ebv, Ebv_2=Ebv, tau_Ebv_1=tau_Ebv, tau_Ebv_2=tau_Ebv,
        gamma_Ebv_1=gamma_Ebv, gamma_Ebv_2=gamma_Ebv
    )

    assert np.allclose(convolution_outputs[:,0], expected_convolution_outputs)
    assert np.allclose(convolution_outputs[:,1], expected_convolution_outputs)
