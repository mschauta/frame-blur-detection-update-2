"""The detector's input, shapes, readouts and weight-file compatibility; no private data. The published weights are
optional."""
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from blurdetect.model import BlurDetector, CELL, KERNEL, SCALE_FLOOR, fingerprint, load_detector, model_input

ROOT = Path(__file__).resolve().parents[1]
V9 = ROOT / "weights" / "v9_base_selfmap_v2_ep019.pt"
V9_SHA256 = "10d6aa1ebe8cb4bf6bb3cd048aaf75edb07fc1bc23a96d1e8a645a87d0f70ba3"
ROUND3 = ROOT / "weights" / "round3_base_p99_ep029.pt"
ROUND3_SHA256 = "9840160f0c337fc1312c2566abf4231be0bd7ee709f65f489fd8f81c65b8d0e6"


def textured(h=48, w=64, seed=0):
    return torch.from_numpy(np.random.default_rng(seed).integers(0, 256, (1, 3, h, w), dtype=np.uint8))


def picture():
    """A deterministic textured picture (192 x 320), its right half blurred by a 25 px horizontal moving average."""
    rng = np.random.default_rng(11)
    y, x = np.mgrid[:192, :320].astype(np.float64)
    img = np.zeros_like(x)
    for _ in range(40):
        fx, fy, ph = rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6), rng.uniform(0, 2 * np.pi)
        img += np.cos(fx * x + fy * y + ph)
    img = 128 + 90 * img / np.abs(img).max()
    pad = np.pad(img, ((0, 0), (12, 12)), mode="edge")
    c = np.cumsum(np.pad(pad, ((0, 0), (1, 0))), axis=1)
    img[:, 160:] = ((c[:, 25:] - c[:, :-25]) / 25)[:, 160:]
    rgb = np.repeat(np.clip(img, 0, 255).round()[..., None], 3, 2).astype(np.uint8)
    return torch.from_numpy(rgb).permute(2, 0, 1)[None]


def test_exact_fingerprint_is_the_float32_formula():
    x = textured()
    lin = x.double() / 255
    lin = torch.where(lin <= 0.04045, lin / 12.92, ((lin + 0.055) / 1.055) ** 2.4)
    y = (lin * torch.tensor([0.2126729, 0.7151522, 0.0721750], dtype=torch.float64).view(1, 3, 1, 1)).sum(1, keepdim=True)
    expected = y - torch.nn.functional.conv2d(torch.nn.functional.pad(y, (1, 1, 1, 1), mode="replicate"),
                                              KERNEL.view(1, 1, 3, 3))
    assert torch.allclose(fingerprint(x, exact=True).double(), expected, atol=1e-6)


def test_bf16_rounded_fingerprint_is_deterministic_and_differs_from_exact():
    x = textured()
    rounded = fingerprint(x)
    assert torch.equal(rounded, fingerprint(x))
    assert not torch.equal(rounded, fingerprint(x, exact=True))
    assert (rounded - fingerprint(x, exact=True)).abs().max() < 0.01


@pytest.mark.parametrize("kind, channels", [("p99", 2), ("rgb", 4), ("gray", 4)])
def test_model_input_channels_and_validity_channel(kind, channels):
    x = textured()
    mask = torch.ones((1, 1, 48, 64))
    mask[..., :, :20] = 0
    out = model_input(x, mask, kind, exact=True)
    assert out.shape == (1, channels, 48, 64)
    assert torch.equal(out[:, -1:], mask)
    assert (out[:, :-1][mask.expand(-1, channels - 1, -1, -1) == 0] == 0).all()


def test_flat_image_uses_the_scale_floor():
    x = torch.full((1, 3, 32, 32), 120, dtype=torch.uint8)
    out = model_input(x, None, "p99", exact=True)
    # No edges: only float32 rounding remains (|d| ~ 1e-8); the floor keeps it near zero instead of blowing it up.
    assert torch.isfinite(out).all() and out[:, 0].abs().max() < 0.01
    assert SCALE_FLOOR == 1e-3


@pytest.fixture(scope="module")
def small():
    torch.manual_seed(0)
    return BlurDetector("p99").eval()


@pytest.fixture(scope="module")
def small_mass():
    torch.manual_seed(1)
    model = BlurDetector("p99", "convnext_small", "fp32", pooling="mass", crop=True).eval()
    with torch.no_grad():                    # centre the random head, so cells of both signs occur
        model.head[-1].bias -= model(textured(96, 128, seed=3))["local"].median()
    return model


def test_forward_shapes_and_cell_validity(small):
    with torch.no_grad():
        out = small(textured(40, 50))
    assert out["local"].shape == (1, 1, 3, 4)                       # ceil(40 / 16), ceil(50 / 16)
    assert torch.isfinite(out["image"]).all()


def test_a_frame_without_a_valid_cell_cannot_be_judged(small):
    with pytest.raises(ValueError, match="no valid"):
        small(textured(10, 10))                                       # one padded cell, under half valid
    with pytest.raises(ValueError, match="no valid"):
        small(textured(32, 32), torch.zeros((1, 1, 32, 32)))


def test_last_half_cell_row_of_1080p_counts():
    # 1080 = 67.5 cells: the padded last row is half valid and counts (>= 0.5) - a documented edge choice.
    valid = torch.nn.functional.pad(torch.ones((1, 1, 1080, CELL)), (0, 0, 0, 8))
    cell_valid = torch.nn.functional.avg_pool2d(valid, CELL, CELL) >= 0.5
    assert cell_valid.shape[-2] == 68 and bool(cell_valid[..., -1, :].all())


def test_mass_readout_is_the_mean_positive_inner_cell_logit(small_mass):
    with torch.no_grad():
        out = small_mass(textured(96, 128, seed=3))
    local = out["local"][0, 0]
    assert local.shape == (6, 8)
    expected = local[1:-1, 1:-1].clamp(min=0).mean()
    assert torch.allclose(out["image"][0], expected, atol=1e-6)
    assert (local > 0).any() and (local < 0).any()                  # both signs occur: the clamp is exercised


def test_mass_readout_ignores_the_outer_ring_and_masked_cells(small_mass):
    x = textured(96, 128, seed=4)
    mask = torch.ones((1, 1, 96, 128))
    mask[..., 16:48, 16:48] = 0                                     # four inner cells masked
    with torch.no_grad():
        out = small_mass(x, mask)
    local = out["local"][0, 0]
    keep = torch.zeros_like(local, dtype=torch.bool)
    keep[1:-1, 1:-1] = True
    keep[1:3, 1:3] = False
    assert torch.allclose(out["image"][0], local[keep].clamp(min=0).mean(), atol=1e-6)


def test_crop_cuts_to_whole_cells_like_training(small_mass):
    x = textured(1080 // 8, 1920 // 8, seed=5)                      # 135 x 240: 8.4 cell rows
    with torch.no_grad():
        full = small_mass(x)
        cut = small_mass(x[..., : 135 // CELL * CELL, :])
    assert full["local"].shape == (1, 1, 8, 15)                      # floor, not ceil
    assert torch.equal(full["local"], cut["local"]) and torch.equal(full["image"], cut["image"])


def test_mass_needs_an_inner_cell(small_mass):
    with pytest.raises(ValueError, match="no valid inner"):
        small_mass(textured(32, 48))                                 # 2 x 3 cells: all on the ring
    with pytest.raises(ValueError, match="smaller than one"):
        small_mass(textured(10, 40))


def test_old_weight_files_keep_their_meaning(tmp_path, small):
    path = tmp_path / "old.pt"
    torch.save({"state_dict": small.state_dict(), "meta": {"input": "p99"}}, path)    # no arch / fingerprint keys
    model, meta = load_detector(path)
    assert not model.exact and model.head[0].in_channels == 384                       # ConvNeXt-Small, bf16-rounded
    assert model.pooling == "lse" and not model.crop
    x = textured(32, 48)
    with torch.no_grad():
        assert torch.equal(model(x)["local"], small(x)["local"])
    base = BlurDetector("p99", "convnext_base", "fp32")
    assert base.exact and base.head[0].in_channels == 512


def test_unknown_pooling_is_refused():
    with pytest.raises(ValueError, match="pooling"):
        BlurDetector("p99", pooling="max")


@pytest.mark.xfail(strict=True, reason="known: the fingerprint of a valid pixel next to a mask is computed from masked "
                   "neighbours, and those pixels enter the p99 scale (README, Limitations); the published weights were "
                   "trained with this behaviour, so it is documented, not changed")
def test_masked_content_does_not_reach_the_valid_region():
    x = torch.full((1, 3, 32, 32), 120, dtype=torch.uint8)
    mask = torch.ones((1, 1, 32, 32))
    mask[..., :16] = 0
    x2 = x.clone()
    x2[..., :16] = 255
    a = model_input(x, mask, "p99", exact=True)
    b = model_input(x2, mask, "p99", exact=True)
    assert torch.allclose(a * mask, b * mask, atol=1e-5)


@pytest.mark.skipif(not V9.is_file(), reason="the Update 2 weight file is a release asset; put it into weights/")
def test_published_update2_weight_file():
    digest = hashlib.sha256(V9.read_bytes()).hexdigest()
    assert digest == V9_SHA256
    model, meta = load_detector(V9)
    assert (meta["arch"], meta["fingerprint"], meta["input"], meta["epoch"]) == ("convnext_base", "fp32", "p99", 19)
    assert (meta["pooling"], meta["crop"], meta["intermediate"]) == ("mass", "cells", False)
    assert "thresholds" not in meta
    assert model.exact and model.pooling == "mass" and model.crop and model.head[0].in_channels == 512
    torch.manual_seed(0)
    with torch.no_grad():
        out = model(picture())
    local = out["local"][0, 0]
    assert local.shape == (12, 20)
    # Reference values of the published file on this picture (CPU, float32); other CPUs and BLAS builds differ by
    # rounding only.
    for (r, c), value in {(1, 1): -2.4773, (5, 4): -3.6491, (5, 15): -5.5884, (10, 18): -4.7467}.items():
        assert abs(float(local[r, c]) - value) < 0.02, (r, c, float(local[r, c]))
    assert abs(float(local.mean()) - (-4.1396)) < 0.02
    assert float(out["image"][0]) == 0.0                             # no inner cell says blur on this picture


@pytest.mark.skipif(not ROUND3.is_file(), reason="the Update 1 weight file is a release asset of Update 1")
def test_published_update1_weight_file_still_loads():
    digest = hashlib.sha256(ROUND3.read_bytes()).hexdigest()
    assert digest == ROUND3_SHA256
    model, meta = load_detector(ROUND3)
    assert (meta["arch"], meta["fingerprint"], meta["epoch"]) == ("convnext_base", "fp32", 29)
    assert set(meta["thresholds"]) == {"r90", "r95", "r98"} and model.pooling == "lse" and not model.crop
    with torch.no_grad():
        out = model(textured(64, 96))
    assert out["local"].shape == (1, 1, 4, 6) and torch.isfinite(out["image"]).all()
