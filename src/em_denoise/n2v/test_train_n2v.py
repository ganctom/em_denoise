"""
Unit tests for the Noise2Void building blocks.

These cover the two pieces of logic that are silently wrong rather than loud when broken:
blind-spot masking (a leak still trains, it just learns the identity) and tiled inference
coverage (a gap still produces an image, it just has a seam).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from em_denoise.n2v.train_n2v import (  # noqa: E402
    apply_blind_spot_mask,
    denoise_array,
    masked_mse,
    split_paths,
    to_unit_float,
)


class _PassThrough(torch.nn.Module):
    """Stand-in for a trained model: returns its input unchanged."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def test_to_unit_float_scales_by_dtype_max():
    image = np.array([[0, 255]], dtype=np.uint8)
    result = to_unit_float(image)
    assert result.dtype == np.float32
    assert result.min() == 0.0 and result.max() == 1.0


@pytest.mark.parametrize("mask_ratio", [0.005, 0.02, 0.1])
def test_mask_marks_expected_number_of_pixels(mask_ratio):
    patch = torch.rand(1, 64, 64)
    _, mask = apply_blind_spot_mask(patch, mask_ratio)
    expected = max(1, round(64 * 64 * mask_ratio))
    assert int(mask.sum().item()) == expected


def test_mask_leaves_unmasked_pixels_untouched():
    patch = torch.rand(1, 64, 64)
    corrupted, mask = apply_blind_spot_mask(patch, 0.05)
    assert torch.equal(corrupted[mask == 0], patch[mask == 0])


def test_masked_pixels_never_keep_their_own_value():
    """
    The blind spot must actually be blind. Uses a patch of unique values so that a masked
    pixel retaining its original value can only mean it was copied from itself.

    Regression test: an earlier version clamped out-of-bounds neighbour offsets, which
    aliased border pixels back onto themselves.
    """
    patch = torch.arange(16 * 16, dtype=torch.float32).reshape(1, 16, 16)
    for _ in range(200):
        corrupted, mask = apply_blind_spot_mask(patch, 0.5)
        leaked = ((corrupted == patch) & (mask == 1)).sum().item()
        assert leaked == 0


def test_masked_mse_ignores_unmasked_error():
    target = torch.zeros(1, 1, 8, 8)
    prediction = torch.ones(1, 1, 8, 8)
    mask = torch.zeros(1, 1, 8, 8)
    mask[0, 0, 0, 0] = 1.0
    # Only the single masked pixel contributes: error 1.0, normalized by mask.sum() == 1.
    assert masked_mse(prediction, target, mask).item() == pytest.approx(1.0)


def test_masked_mse_is_independent_of_mask_count():
    """Loss magnitude must not scale with mask_ratio, else the LR changes with it."""
    target = torch.zeros(1, 1, 16, 16)
    prediction = torch.full((1, 1, 16, 16), 2.0)

    sparse = torch.zeros(1, 1, 16, 16)
    sparse[0, 0, :2, :2] = 1.0
    dense = torch.ones(1, 1, 16, 16)

    assert masked_mse(prediction, target, sparse).item() == pytest.approx(
        masked_mse(prediction, target, dense).item()
    )


def test_split_paths_is_deterministic_and_disjoint():
    paths = [f"img_{i}.tif" for i in range(20)]
    train_a, val_a = split_paths(paths, 0.2, seed=7)
    train_b, val_b = split_paths(paths, 0.2, seed=7)

    assert (train_a, val_a) == (train_b, val_b)
    assert set(train_a).isdisjoint(val_a)
    assert len(train_a) + len(val_a) == len(paths)
    assert len(val_a) == 4


@pytest.mark.parametrize("shape", [(128, 128), (300, 200), (100, 100)])
def test_tiled_inference_covers_image_and_preserves_signal(shape):
    """
    A pass-through model must reproduce its input, which only holds if every pixel is
    covered by at least one tile and the Hann blend weights normalize to 1.
    """
    height, width = shape
    image = (np.random.default_rng(0).random((height, width)) * 255).astype(np.uint8)

    result = denoise_array(
        _PassThrough(), image, torch.device("cpu"), tile_size=64, overlap=16
    )

    assert result.shape == (height, width)
    assert result.dtype == np.uint8
    # Round-trip through uint8 quantization allows a tolerance of one level.
    assert np.abs(result.astype(int) - image.astype(int)).max() <= 1


def test_denoise_array_rejects_images_smaller_than_one_tile():
    with pytest.raises(ValueError):
        denoise_array(_PassThrough(), np.zeros((4, 4), np.uint8), torch.device("cpu"))
