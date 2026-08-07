# src/em_denoise_refactor/tests/test_cosdd.py

import torch
from em_denoise_refactor.models.factory import build_model
from em_denoise_refactor.models.cosdd import COSDDModel


def test_cosdd_model_build_and_forward():
    model = build_model(
        model_type="cosdd",
        in_channels=1,
        out_channels=1,
        crop_size=(64, 64),
        s_code_channels=32,
        number_layers=4,
        number_components=3,
    )
    assert isinstance(model, COSDDModel)

    x = torch.randn(2, 1, 64, 64)
    s_hat, (loss, metrics) = model(x)

    assert s_hat.shape == x.shape
    assert isinstance(loss, torch.Tensor)
    assert "loss_total" in metrics
    assert "loss_elbo" in metrics
    assert "loss_kl" in metrics
    assert "loss_nll" in metrics
    assert "loss_s_decoder" in metrics

    # Fast inference / denoise test
    clean_hat = model.denoise(x)
    assert clean_hat.shape == x.shape
