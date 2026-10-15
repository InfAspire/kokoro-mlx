# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Soroush Yousefpour

"""Regression guards for reduced-precision numerics (NaN / silent audio, issue #2).

None of these tests need the model weights: they build small modules with random
weights and feed them float16 / bfloat16 data shaped like real decoder activations.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from kokoro_mlx.istftnet import AdaIN1d, Generator, SineGen, _reflect_pad_1d, instance_norm1d, iSTFT
from kokoro_mlx.model import _load_safetensors
from kokoro_mlx.modules import AdaLayerNorm, LayerNorm, normalize
from kokoro_mlx.voices import VoiceManager

HALF_DTYPES = [mx.float16, mx.bfloat16]


def _np(x: mx.array) -> np.ndarray:
    return np.array(x.astype(mx.float32))


def _large_activations(shape, dtype) -> mx.array:
    """Values around 300 with spread 50: x**2 overflows float16 (max 65504)."""
    mx.random.seed(0)
    return (300.0 + 50.0 * mx.random.normal(shape)).astype(dtype)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", HALF_DTYPES)
def test_normalize_half_precision_is_finite_and_matches_float32(dtype):
    x = _large_activations((2, 8, 64), dtype)
    out = normalize(x, axis=-1)
    assert out.dtype == dtype
    ref = normalize(x.astype(mx.float32), axis=-1)
    assert np.all(np.isfinite(_np(out)))
    np.testing.assert_allclose(_np(out), _np(ref), atol=5e-2)


@pytest.mark.parametrize("dtype", HALF_DTYPES)
def test_instance_norm1d_half_precision_is_finite(dtype):
    out = instance_norm1d(_large_activations((1, 16, 128), dtype))
    arr = _np(out)
    assert np.all(np.isfinite(arr))
    np.testing.assert_allclose(arr.mean(axis=-1), 0.0, atol=5e-2)
    np.testing.assert_allclose(arr.std(axis=-1), 1.0, atol=5e-2)


@pytest.mark.parametrize("dtype", HALF_DTYPES)
def test_layernorm_half_precision_is_finite(dtype):
    out = LayerNorm(16)(_large_activations((1, 16, 32), dtype))
    assert np.all(np.isfinite(_np(out)))


@pytest.mark.parametrize("dtype", HALF_DTYPES)
def test_ada_layernorm_half_precision_is_finite(dtype):
    out = AdaLayerNorm(8, 16)(_large_activations((1, 32, 16), dtype), mx.random.normal((1, 8)))
    assert np.all(np.isfinite(_np(out)))


@pytest.mark.parametrize("dtype", HALF_DTYPES)
def test_adain1d_half_precision_is_finite(dtype):
    out = AdaIN1d(8, 16)(_large_activations((1, 16, 64), dtype), mx.random.normal((1, 8)))
    assert np.all(np.isfinite(_np(out)))


# ---------------------------------------------------------------------------
# Source module and STFT
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", HALF_DTYPES)
def test_sinegen_accumulates_phase_in_float32(dtype):
    gen = SineGen(samp_rate=24000, upsample_scale=300, harmonic_num=8, voiced_threshold=10)
    f0 = mx.full((1, 300 * 240, 1), 220.0).astype(dtype)  # 3 s of voiced audio
    sines, uv, _ = gen(f0)
    assert sines.dtype == mx.float32
    arr = _np(sines)
    assert np.all(np.isfinite(arr))
    assert np.abs(arr).max() > 0.05  # not silent
    assert np.all(_np(uv) == 1.0)


def test_reflect_pad_matches_numpy():
    x = mx.arange(12, dtype=mx.float32).reshape(1, 2, 6)
    for left, right in [(1, 0), (0, 3), (5, 5), (2, 4)]:
        expected = np.pad(np.array(x), ((0, 0), (0, 0), (left, right)), mode="reflect")
        np.testing.assert_array_equal(np.array(_reflect_pad_1d(x, left, right)), expected)


def test_reflect_pad_rejects_too_large_padding():
    with pytest.raises(ValueError):
        _reflect_pad_1d(mx.zeros((1, 3)), 3, 0)


def test_stft_transform_matches_reflect_padded_numpy_stft():
    n_fft, hop = 20, 5
    stft = iSTFT(filter_length=n_fft, hop_length=hop, win_length=n_fft)
    rng = np.random.default_rng(0)
    signal = rng.standard_normal(200).astype(np.float32)
    mag, _ = stft.transform(mx.array(signal[None]))

    padded = np.pad(signal, n_fft // 2, mode="reflect")
    window = np.hanning(n_fft + 1)[:-1]
    frames = np.stack([padded[i : i + n_fft] * window for i in range(0, len(padded) - n_fft + 1, hop)])
    expected = np.abs(np.fft.rfft(frames, axis=-1)).T
    np.testing.assert_allclose(np.array(mag[0]), expected, atol=1e-3)


@pytest.mark.parametrize("dtype", HALF_DTYPES)
def test_generator_half_precision_inputs_produce_finite_audio(dtype):
    mx.random.seed(0)
    gen = Generator(
        style_dim=8,
        resblock_kernel_sizes=[3],
        upsample_rates=[10, 6],
        upsample_initial_channel=16,
        resblock_dilation_sizes=[[1, 3, 5]],
        upsample_kernel_sizes=[20, 12],
        gen_istft_n_fft=20,
        gen_istft_hop_size=5,
    )
    frames = 6
    # Unit-scale features: with random weights, large inputs would overflow the
    # spectrogram exp() in any precision, which is not what this test guards.
    x = mx.random.normal((1, 16, frames)).astype(dtype)
    s = mx.random.normal((1, 8)).astype(dtype)
    f0 = mx.full((1, frames), 180.0).astype(dtype)
    audio = gen(x, s, f0)
    assert audio.dtype == mx.float32
    assert audio.shape == (1, 1, frames * 300)
    assert np.all(np.isfinite(np.array(audio)))


# ---------------------------------------------------------------------------
# Weight and voice loading
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", HALF_DTYPES)
def test_load_safetensors_upcasts_reduced_precision(tmp_path, dtype):
    path = tmp_path / "weights.safetensors"
    mx.save_safetensors(
        str(path),
        {"w": mx.ones((4, 3), dtype=dtype), "ids": mx.arange(3, dtype=mx.int32)},
    )
    weights = _load_safetensors(path)
    assert weights["w"].dtype == mx.float32
    assert weights["ids"].dtype == mx.int32
    np.testing.assert_array_equal(np.array(weights["w"]), np.ones((4, 3), dtype=np.float32))


@pytest.mark.parametrize("dtype", HALF_DTYPES)
def test_voice_manager_upcasts_reduced_precision(tmp_path, dtype):
    voices = tmp_path / "voices"
    voices.mkdir()
    mx.save_safetensors(str(voices / "af_test.safetensors"), {"voice": mx.ones((510, 1, 256), dtype=dtype)})
    manager = VoiceManager(tmp_path)
    assert manager.list_voices() == ["af_test"]
    voice = manager.load_voice("af_test")
    assert voice.dtype == mx.float32
    assert voice.shape == (510, 1, 256)
