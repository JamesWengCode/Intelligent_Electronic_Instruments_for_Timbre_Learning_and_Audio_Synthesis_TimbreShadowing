"""
train_timbre.py

Learn a single instrument timbre from one recorded note (DDSPHarmonicSynthesizer):
a per-frame amplitude envelope, harmonic distribution, brightness decay,
attack transient noise, and continuous broadband noise, all directly
optimized against the target audio via gradient descent -- no encoder
network, no generalization across recordings, one fit per note.

Run from project root:

python3 src/train_timbre.py --target <audio_path> --out_dir <output_dir>

Output (in --out_dir):

learned_params.json   (learned envelopes/profiles, read by render_notes.py)
output.wav             (reconstruction of the target at its own pitch)
target_normalized.wav  (the target audio as actually seen during training)
*.png                  (diagnostic plots: loss curve, spectrograms, envelopes)
"""

import argparse
import json
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

BASE_DIR = Path(__file__).resolve().parent


def load_audio(path, sr=16000, max_seconds=4.0):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Cannot find target audio: {path}")

    y, _ = librosa.load(path, sr=sr, mono=True)

    max_len = int(sr * max_seconds)

    if len(y) > max_len:
        y = y[:max_len]

    if len(y) < max_len:
        y = np.pad(y, (0, max_len - len(y)))

    peak = np.max(np.abs(y)) + 1e-8
    y = y / peak * 0.8

    return y.astype(np.float32)


def estimate_f0(y, sr):
    try:
        f0 = librosa.yin(
            y,
            fmin=40,
            fmax=1200,
            sr=sr,
            frame_length=2048,
            hop_length=256,
        )

        f0 = f0[np.isfinite(f0)]
        f0 = f0[(f0 > 40) & (f0 < 1200)]

        if len(f0) == 0:
            return 110.0

        return float(np.median(f0))

    except Exception:
        return 110.0


def estimate_auto_harmonic_cutoff(
    y,
    sr,
    f0,
    db_threshold=-72.0,
    min_cutoff=900.0,
    max_cutoff=5000.0,
):
    """
    Automatically estimate useful harmonic range from target audio.

    Lower db_threshold  -> keep more high harmonics.
    Higher db_threshold -> cut more high harmonics.
    """
    n_fft = 4096
    hop_length = 512

    spec = np.abs(
        librosa.stft(
            y,
            n_fft=n_fft,
            hop_length=hop_length,
            center=True,
        )
    )

    mean_spec = np.mean(spec, axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    mean_db = librosa.amplitude_to_db(mean_spec, ref=np.max)

    nyquist = sr / 2
    max_possible = min(max_cutoff, nyquist * 0.95)
    max_h = int(max_possible // f0)

    useful_harmonics = []

    for h in range(1, max_h + 1):
        harmonic_freq = h * f0
        bandwidth = max(20.0, f0 * 0.18)

        mask = (freqs >= harmonic_freq - bandwidth) & (
            freqs <= harmonic_freq + bandwidth
        )

        if not np.any(mask):
            continue

        band_energy_db = np.max(mean_db[mask])

        if band_energy_db > db_threshold:
            useful_harmonics.append(h)

    if len(useful_harmonics) == 0:
        return float(min_cutoff)

    last_h = max(useful_harmonics)

    cutoff = (last_h + 1) * f0
    cutoff = max(cutoff, min_cutoff)
    cutoff = min(cutoff, max_possible)

    return float(cutoff)


def estimate_target_harmonic_profile(y, sr, f0, n_harmonics):
    """
    Estimate average target harmonic energy distribution.

    Returns:
        harmonic_profile: [n_harmonics], normalized to sum to 1.
    """
    n_fft = 4096
    hop_length = 512

    spec = np.abs(
        librosa.stft(
            y,
            n_fft=n_fft,
            hop_length=hop_length,
            center=True,
        )
    )

    mean_spec = np.mean(spec, axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    harmonic_energies = []

    for h in range(1, n_harmonics + 1):
        harmonic_freq = h * f0
        bandwidth = max(20.0, f0 * 0.18)

        mask = (freqs >= harmonic_freq - bandwidth) & (
            freqs <= harmonic_freq + bandwidth
        )

        if not np.any(mask):
            harmonic_energies.append(1e-8)
            continue

        energy = np.max(mean_spec[mask])
        harmonic_energies.append(float(energy))

    harmonic_energies = np.array(harmonic_energies, dtype=np.float32)
    harmonic_energies = harmonic_energies + 1e-8
    harmonic_profile = harmonic_energies / np.sum(harmonic_energies)

    return harmonic_profile


def estimate_target_harmonic_profile_frames(y, sr, f0, n_harmonics, n_frames):
    """
    Estimate time-varying target harmonic profile.

    Returns:
        target_profiles: [n_frames, n_harmonics], normalized per frame.
        frame_weights: [n_frames], derived from target RMS envelope.
    """
    n_fft = 2048
    hop_length = 256

    spec = np.abs(
        librosa.stft(
            y,
            n_fft=n_fft,
            hop_length=hop_length,
            center=True,
        )
    )

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    n_spec_frames = spec.shape[1]

    harmonic_energy = np.zeros((n_spec_frames, n_harmonics), dtype=np.float32)

    for h in range(1, n_harmonics + 1):
        harmonic_freq = h * f0
        bandwidth = max(20.0, f0 * 0.18)

        mask = (freqs >= harmonic_freq - bandwidth) & (
            freqs <= harmonic_freq + bandwidth
        )

        if not np.any(mask):
            harmonic_energy[:, h - 1] = 1e-8
        else:
            harmonic_energy[:, h - 1] = np.max(spec[mask, :], axis=0)

    harmonic_energy = harmonic_energy + 1e-8
    harmonic_profile = harmonic_energy / (
        np.sum(harmonic_energy, axis=1, keepdims=True) + 1e-8
    )

    rms = librosa.feature.rms(
        y=y,
        frame_length=n_fft,
        hop_length=hop_length,
        center=True,
    )[0]

    rms = rms[:n_spec_frames]
    rms = rms / (np.max(rms) + 1e-8)

    source_x = np.linspace(0.0, 1.0, n_spec_frames)
    target_x = np.linspace(0.0, 1.0, n_frames)

    target_profiles = np.zeros((n_frames, n_harmonics), dtype=np.float32)

    for h in range(n_harmonics):
        target_profiles[:, h] = np.interp(
            target_x,
            source_x,
            harmonic_profile[:, h],
        )

    target_profiles = target_profiles + 1e-8
    target_profiles = target_profiles / (
        np.sum(target_profiles, axis=1, keepdims=True) + 1e-8
    )

    frame_weights = np.interp(target_x, source_x, rms).astype(np.float32)
    frame_weights = frame_weights / (np.max(frame_weights) + 1e-8)

    return target_profiles.astype(np.float32), frame_weights.astype(np.float32)


def make_odd(k):
    k = int(k)

    if k < 1:
        return 1

    if k % 2 == 0:
        k += 1

    return k


def smooth_1d(x, kernel_size):
    """
    x: [T]
    return: [T]
    """
    kernel_size = make_odd(kernel_size)

    if kernel_size <= 1:
        return x

    pad = kernel_size // 2
    x = x[None, None, :]
    x = F.pad(x, (pad, pad), mode="replicate")
    x = F.avg_pool1d(x, kernel_size=kernel_size, stride=1)

    return x[0, 0]


def smooth_2d_time(x, kernel_size):
    """
    x: [T, H]
    return: [T, H]
    """
    kernel_size = make_odd(kernel_size)

    if kernel_size <= 1:
        return x

    pad = kernel_size // 2
    x = x.T[None, :, :]
    x = F.pad(x, (pad, pad), mode="replicate")
    x = F.avg_pool1d(x, kernel_size=kernel_size, stride=1)

    return x[0].T


def upsample_1d(x, target_len):
    """
    x: [T]
    return: [target_len]
    """
    x = x[None, None, :]
    x = F.interpolate(x, size=target_len, mode="linear", align_corners=True)
    return x[0, 0]


def upsample_2d_time(x, target_len):
    """
    x: [T, H]
    return: [target_len, H]
    """
    x = x.T[None, :, :]
    x = F.interpolate(x, size=target_len, mode="linear", align_corners=True)
    return x[0].T


def stft_mag(x, fft_size):
    hop = fft_size // 4
    window = torch.hann_window(fft_size, device=x.device)

    spec = torch.stft(
        x,
        n_fft=fft_size,
        hop_length=hop,
        win_length=fft_size,
        window=window,
        return_complex=True,
        center=True,
    )

    return torch.abs(spec) + 1e-7


def multi_scale_stft_loss(x, y, fft_sizes=(2048, 1024, 512, 256, 128), log_mag_weight=1.0):
    # spectral_convergence (linear-magnitude) is naturally dominated by
    # low-frequency error; log_mag_loss is scale-invariant and corrects for
    # that. log_mag_weight defaults to 1.0 (no behavior change) -- raise it
    # to weight high-frequency accuracy more.
    total = 0.0

    for fft_size in fft_sizes:
        x_mag = stft_mag(x, fft_size)
        y_mag = stft_mag(y, fft_size)

        spectral_convergence = torch.norm(y_mag - x_mag, p="fro") / (
            torch.norm(y_mag, p="fro") + 1e-7
        )

        log_mag_loss = torch.mean(torch.abs(torch.log(x_mag) - torch.log(y_mag)))

        total = total + spectral_convergence + log_mag_weight * log_mag_loss

    return total


def audio_rms_envelope(x, n_frames):
    """
    Differentiable audio envelope from waveform.

    x: [samples]
    return: [n_frames], normalized
    """
    n_samples = x.shape[0]
    kernel_size = max(64, n_samples // (n_frames * 2))
    kernel_size = min(kernel_size, n_samples)

    if kernel_size % 2 == 0:
        kernel_size += 1

    pad = kernel_size // 2

    x_abs = torch.abs(x)[None, None, :]
    x_abs = F.pad(x_abs, (pad, pad), mode="reflect")
    env = F.avg_pool1d(x_abs, kernel_size=kernel_size, stride=kernel_size // 2)

    env = F.interpolate(env, size=n_frames, mode="linear", align_corners=True)
    env = env[0, 0]

    env = env / (torch.max(env) + 1e-8)

    return env


def smoothness_1d(x):
    return torch.mean(torch.abs(x[1:] - x[:-1]))


def smoothness_2d_time(x):
    return torch.mean(torch.abs(x[1:, :] - x[:-1, :]))


def curvature_1d(x):
    """
    Penalize zig-zag movement.
    """
    if x.shape[0] < 3:
        return torch.tensor(0.0, device=x.device)

    return torch.mean(torch.abs(x[2:] - 2 * x[1:-1] + x[:-2]))


def curvature_2d_time(x):
    """
    Penalize zig-zag movement over time.
    x: [T, H]
    """
    if x.shape[0] < 3:
        return torch.tensor(0.0, device=x.device)

    return torch.mean(torch.abs(x[2:, :] - 2 * x[1:-1, :] + x[:-2, :]))


def amp_to_impulse_response(amp, target_size):
    """
    Frequency-sampling method: turn a per-frame nonnegative magnitude
    spectrum into a windowed, causal-centered real FIR impulse response.

    amp: [..., n_mags] nonnegative magnitudes (rfft bins, DC..Nyquist)
    return: [..., target_size]
    """
    impulse = torch.fft.irfft(amp.to(torch.complex64))
    filter_size = impulse.shape[-1]

    impulse = torch.roll(impulse, filter_size // 2, dims=-1)
    window = torch.hann_window(
        filter_size, periodic=False, device=impulse.device, dtype=impulse.dtype
    )
    impulse = impulse * window

    if target_size > filter_size:
        impulse = F.pad(impulse, (0, target_size - filter_size))

    impulse = torch.roll(impulse, -filter_size // 2, dims=-1)
    return impulse


def fft_convolve(signal, kernel):
    """
    signal, kernel: [..., n] (same trailing length n)
    Zero-padded FFT convolution; returns the causal linear-convolution
    result truncated back to length n.
    """
    n = signal.shape[-1]
    signal = F.pad(signal, (0, n))
    kernel = F.pad(kernel, (n, 0))

    output = torch.fft.irfft(
        torch.fft.rfft(signal) * torch.fft.rfft(kernel), n=signal.shape[-1]
    )
    return output[..., n:]


def log_harmonic_profile_loss(learned_profile, target_profile):
    """
    Penalize relative harmonic profile differences.
    """
    eps = 1e-5

    learned_log = torch.log(learned_profile + eps)
    target_log = torch.log(target_profile + eps)

    return torch.mean(torch.abs(learned_log - target_log))


def harmonic_over_penalty(learned_profile, target_profile):
    """
    Penalize harmonics that are stronger than the target.
    """
    eps = 1e-5

    learned_log = torch.log(learned_profile + eps)
    target_log = torch.log(target_profile + eps)

    over = torch.relu(learned_log - target_log)

    return torch.mean(over**2)


def harmonic_max_over_penalty(learned_profile, target_profile):
    """
    Penalize the single worst harmonic that is stronger than the target.
    """
    eps = 1e-5

    learned_log = torch.log(learned_profile + eps)
    target_log = torch.log(target_profile + eps)

    over = torch.relu(learned_log - target_log)

    return torch.max(over**2)


def time_harmonic_profile_loss(
    learned_frames,
    target_frames,
    frame_weights,
):
    """
    Match harmonic profile over time.

    This is important for plucked/decaying sounds where high harmonics
    mainly exist at the attack and disappear later.
    """
    eps = 1e-5

    l1 = torch.abs(learned_frames - target_frames)

    learned_log = torch.log(learned_frames + eps)
    target_log = torch.log(target_frames + eps)
    log_l1 = torch.abs(learned_log - target_log)

    weights = frame_weights[:, None]

    return torch.mean(weights * l1), torch.mean(weights * log_l1)


class DDSPHarmonicSynthesizer(torch.nn.Module):
    def __init__(
        self,
        sr,
        n_samples,
        f0,
        n_harmonics=40,
        n_frames=150,
        max_harmonic_freq=1800.0,
        residual_strength=0.20,
        brightness_strength=2.0,
        amp_smooth_kernel=7,
        harmonic_smooth_kernel=9,
        brightness_smooth_kernel=9,
        use_transient_noise=True,
        transient_decay=8.0,
        noise_init=-10.0,
        use_broadband_noise=True,
        n_noise_mags=65,
    ):
        super().__init__()

        self.sr = sr
        self.n_samples = n_samples
        self.f0 = float(f0)
        self.n_frames = n_frames
        self.residual_strength = residual_strength
        self.brightness_strength = brightness_strength
        self.amp_smooth_kernel = amp_smooth_kernel
        self.harmonic_smooth_kernel = harmonic_smooth_kernel
        self.brightness_smooth_kernel = brightness_smooth_kernel
        self.use_transient_noise = use_transient_noise
        self.transient_decay = transient_decay
        self.use_broadband_noise = use_broadband_noise
        self.n_noise_mags = n_noise_mags
        self.noise_hop = max(8, n_samples // n_frames)

        nyquist = sr / 2
        max_allowed_freq = min(max_harmonic_freq, nyquist * 0.95)
        max_harmonics_by_freq = int(max_allowed_freq // self.f0)

        self.n_harmonics = max(1, min(n_harmonics, max_harmonics_by_freq))

        t = torch.arange(n_samples).float() / sr
        harmonic_numbers = torch.arange(1, self.n_harmonics + 1).float()

        phase = 2.0 * np.pi * self.f0 * harmonic_numbers[:, None] * t[None, :]
        sine_bank = torch.sin(phase)

        self.register_buffer("sine_bank", sine_bank)
        self.register_buffer("harmonic_numbers", harmonic_numbers)

        if self.n_harmonics == 1:
            harmonic_index = torch.zeros(1)
        else:
            harmonic_index = (harmonic_numbers - 1.0) / (self.n_harmonics - 1.0)

        self.register_buffer("harmonic_index", harmonic_index)

        fixed_noise = torch.randn(n_samples) * 0.02
        fixed_noise = fixed_noise - torch.mean(fixed_noise)
        self.register_buffer("fixed_noise", fixed_noise)

        frame_pos = torch.linspace(0.0, 1.0, n_frames)
        transient_mask = torch.exp(-transient_decay * frame_pos)
        self.register_buffer("transient_mask", transient_mask)

        # Smoothed amplitude envelope.
        self.amp_logits_raw = torch.nn.Parameter(torch.ones(n_frames) * -2.0)

        # Stable base harmonic distribution.
        base_slope = -torch.linspace(0.0, 4.0, self.n_harmonics)
        base_slope = base_slope + torch.randn(self.n_harmonics) * 0.01
        self.harmonic_base_logits = torch.nn.Parameter(base_slope)

        # Limited time-varying residual.
        self.harmonic_residual_logits = torch.nn.Parameter(
            torch.randn(n_frames, self.n_harmonics) * 0.01
        )

        # Time-varying brightness/darkness.
        # Larger brightness value means stronger high-harmonic damping.
        brightness_init = torch.linspace(-2.5, 0.5, n_frames)
        self.brightness_logits_raw = torch.nn.Parameter(brightness_init)

        # Attack transient noise.
        transient_init = torch.linspace(-1.0, -9.0, n_frames)
        self.transient_noise_logits_raw = torch.nn.Parameter(transient_init)
        self.transient_gain_logit = torch.nn.Parameter(torch.tensor(-2.0))

        # Continuous time-varying filtered noise (frequency-sampling method).
        # Unlike the transient burst above, this runs for the full note and
        # is meant to capture the recording's ever-present breath/bow/air
        # noise floor that pure harmonic synthesis can't reproduce.
        broadband_noise = torch.randn(self.noise_hop * n_frames)
        broadband_noise = broadband_noise - torch.mean(broadband_noise)
        self.register_buffer("broadband_noise", broadband_noise)

        self.noise_mag_logits = torch.nn.Parameter(
            torch.full((n_frames, n_noise_mags), -6.0)
        )
        self.noise_energy_logits_raw = torch.nn.Parameter(
            torch.full((n_frames,), -6.0)
        )
        self.noise_gain_logit = torch.nn.Parameter(torch.tensor(-2.0))

        # Output gain.
        self.output_gain_logit = torch.nn.Parameter(torch.tensor(-1.5))

    def forward(self):
        amp_logits = smooth_1d(self.amp_logits_raw, self.amp_smooth_kernel)
        amp_frames = torch.sigmoid(amp_logits)
        amp = upsample_1d(amp_frames, self.n_samples)

        residual = self.residual_strength * torch.tanh(self.harmonic_residual_logits)

        harmonic_logits_frames = self.harmonic_base_logits[None, :] + residual

        # Brightness curve: makes high harmonics decay over time.
        brightness_logits = smooth_1d(
            self.brightness_logits_raw,
            self.brightness_smooth_kernel,
        )
        brightness_frames = F.softplus(brightness_logits)

        high_harmonic_damping = (
            self.brightness_strength
            * brightness_frames[:, None]
            * self.harmonic_index[None, :]
        )

        harmonic_logits_frames = harmonic_logits_frames - high_harmonic_damping

        harmonic_logits_frames = smooth_2d_time(
            harmonic_logits_frames,
            self.harmonic_smooth_kernel,
        )

        harmonic_distribution_frames = torch.softmax(harmonic_logits_frames, dim=-1)

        harmonic_distribution = upsample_2d_time(
            harmonic_distribution_frames,
            self.n_samples,
        )

        harmonic_distribution = harmonic_distribution / (
            harmonic_distribution.sum(dim=-1, keepdim=True) + 1e-8
        )

        sine_bank = self.sine_bank.T

        harmonic_audio = torch.sum(
            harmonic_distribution * sine_bank,
            dim=-1,
        )

        gain = F.softplus(self.output_gain_logit)

        audio = gain * amp * harmonic_audio

        transient_env_frames = torch.zeros_like(amp_frames)
        transient_gain = F.softplus(self.transient_gain_logit)

        if self.use_transient_noise:
            transient_logits = smooth_1d(
                self.transient_noise_logits_raw,
                self.amp_smooth_kernel,
            )

            transient_env_frames = torch.sigmoid(transient_logits)
            transient_env_frames = transient_env_frames * self.transient_mask

            transient_env = upsample_1d(transient_env_frames, self.n_samples)

            audio = audio + transient_gain * transient_env * self.fixed_noise

        noise_mag_frames = torch.zeros(
            self.n_frames, self.n_noise_mags, device=amp_frames.device
        )
        noise_energy_frames = torch.zeros_like(amp_frames)
        noise_gain = torch.tensor(0.0, device=amp_frames.device)

        if self.use_broadband_noise:
            noise_mag_frames = F.softplus(
                smooth_2d_time(self.noise_mag_logits, self.harmonic_smooth_kernel)
            )
            noise_energy_frames = torch.sigmoid(
                smooth_1d(self.noise_energy_logits_raw, self.amp_smooth_kernel)
            )
            noise_gain = F.softplus(self.noise_gain_logit)

            hop = self.noise_hop
            impulse = amp_to_impulse_response(noise_mag_frames, hop)

            noise_blocks = self.broadband_noise.reshape(self.n_frames, hop)
            filtered_blocks = fft_convolve(noise_blocks, impulse)

            block_rms = torch.sqrt(
                torch.mean(filtered_blocks**2, dim=-1, keepdim=True) + 1e-8
            )
            filtered_blocks = filtered_blocks / block_rms

            filtered_noise = filtered_blocks.reshape(-1)
            if filtered_noise.shape[0] < self.n_samples:
                filtered_noise = F.pad(
                    filtered_noise, (0, self.n_samples - filtered_noise.shape[0])
                )
            else:
                filtered_noise = filtered_noise[: self.n_samples]

            noise_energy = upsample_1d(noise_energy_frames, self.n_samples)

            audio = audio + noise_gain * noise_energy * filtered_noise

        audio = torch.clamp(audio, -1.0, 1.0)

        return audio, {
            "amp_frames": amp_frames,
            "harmonic_distribution_frames": harmonic_distribution_frames,
            "harmonic_logits_frames": harmonic_logits_frames,
            "harmonic_residual": residual,
            "brightness_frames": brightness_frames,
            "transient_env_frames": transient_env_frames,
            "gain": gain,
            "transient_gain": transient_gain,
            "noise_mag_frames": noise_mag_frames,
            "noise_energy_frames": noise_energy_frames,
            "noise_gain": noise_gain,
        }


def save_wav(path, audio, sr):
    audio = np.asarray(audio, dtype=np.float32)

    peak = np.max(np.abs(audio)) + 1e-8
    if peak > 1.0:
        audio = audio / peak * 0.95

    sf.write(path, audio, sr)


def plot_loss(losses, out_dir):
    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=150)
    plt.close()


def plot_spectrogram(audio, sr, title, path):
    spec = librosa.amplitude_to_db(
        np.abs(librosa.stft(audio, n_fft=1024, hop_length=256)),
        ref=np.max,
    )

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        spec,
        sr=sr,
        hop_length=256,
        x_axis="time",
        y_axis="hz",
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.ylim(0, 8000)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_mean_harmonics(harmonic_frames, f0, out_dir):
    mean_weights = harmonic_frames.mean(axis=0)
    freqs = np.arange(1, len(mean_weights) + 1) * f0

    plt.figure(figsize=(10, 4))
    plt.bar(freqs, mean_weights, width=f0 * 0.7)
    plt.title("Mean Learned Harmonic Distribution")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Mean Weight")
    plt.tight_layout()
    plt.savefig(out_dir / "mean_harmonic_distribution.png", dpi=150)
    plt.close()


def plot_target_vs_learned_profile(target_profile, learned_profile, f0, out_dir):
    freqs = np.arange(1, len(target_profile) + 1) * f0

    plt.figure(figsize=(10, 4))
    plt.plot(freqs, target_profile, marker="o", label="Target")
    plt.plot(freqs, learned_profile, marker="o", label="Learned")
    plt.title("Target vs Learned Harmonic Profile")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Normalized Weight")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "target_vs_learned_harmonic_profile.png", dpi=150)
    plt.close()


def plot_harmonic_heatmap(harmonic_frames, f0, out_dir):
    data = harmonic_frames.T

    plt.figure(figsize=(10, 5))
    plt.imshow(
        data,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )
    plt.colorbar(label="Weight")
    plt.title("Time-Varying Harmonic Distribution")
    plt.xlabel("Frame")
    plt.ylabel("Harmonic Frequency")

    n_harmonics = data.shape[0]
    yticks = np.arange(n_harmonics)
    ylabels = [f"{(i + 1) * f0:.0f}" for i in range(n_harmonics)]
    plt.yticks(yticks, ylabels)

    plt.tight_layout()
    plt.savefig(out_dir / "harmonic_distribution_heatmap.png", dpi=150)
    plt.close()


def plot_amp_envelope(amp_frames, target_env, out_dir):
    plt.figure(figsize=(10, 4))
    plt.plot(amp_frames, label="Learned amplitude")
    plt.plot(target_env, label="Target envelope")
    plt.title("Amplitude Envelope")
    plt.xlabel("Frame")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "amplitude_envelope.png", dpi=150)
    plt.close()


def plot_brightness(brightness_frames, out_dir):
    plt.figure(figsize=(10, 4))
    plt.plot(brightness_frames)
    plt.title("Learned Brightness Damping")
    plt.xlabel("Frame")
    plt.ylabel("Brightness Damping")
    plt.tight_layout()
    plt.savefig(out_dir / "brightness_damping.png", dpi=150)
    plt.close()


def plot_transient(transient_env_frames, out_dir):
    plt.figure(figsize=(10, 4))
    plt.plot(transient_env_frames)
    plt.title("Learned Transient Noise Envelope")
    plt.xlabel("Frame")
    plt.ylabel("Transient Noise")
    plt.tight_layout()
    plt.savefig(out_dir / "transient_noise_envelope.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target",
        type=str,
        default=str(BASE_DIR / "audio" / "target.wav"),
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(BASE_DIR / "outputs"),
    )

    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--seconds", type=float, default=4.0)

    parser.add_argument("--steps", type=int, default=9000)
    parser.add_argument("--lr", type=float, default=0.003)

    # Early stopping keeps the best-loss parameters and stops once the loss
    # stops improving meaningfully, rather than always running the full
    # step count.
    parser.add_argument("--disable_early_stopping", action="store_true")
    parser.add_argument("--early_stop_min_steps", type=int, default=1500)
    parser.add_argument("--early_stop_check_interval", type=int, default=100)
    parser.add_argument("--early_stop_patience", type=int, default=3)
    parser.add_argument(
        "--early_stop_min_relative_improvement",
        type=float,
        default=0.005,
        help="Required relative loss improvement at each check (0.001 = 0.1%%).",
    )

    parser.add_argument("--f0", type=str, default="auto")
    parser.add_argument("--harmonics", type=int, default=40)

    parser.add_argument("--max_harmonic_freq", type=str, default="auto")
    parser.add_argument("--auto_harmonic_db", type=float, default=-72.0)
    parser.add_argument("--auto_min_cutoff", type=float, default=900.0)
    parser.add_argument("--auto_max_cutoff", type=float, default=8000.0)

    parser.add_argument("--n_frames", type=int, default=150)

    parser.add_argument("--residual_strength", type=float, default=0.20)
    parser.add_argument("--brightness_strength", type=float, default=2.0)

    parser.add_argument("--amp_smooth_kernel", type=int, default=7)
    parser.add_argument("--harmonic_smooth_kernel", type=int, default=9)
    parser.add_argument("--brightness_smooth_kernel", type=int, default=9)

    parser.add_argument("--disable_transient_noise", action="store_true")
    parser.add_argument("--transient_decay", type=float, default=8.0)

    parser.add_argument("--disable_broadband_noise", action="store_true")
    parser.add_argument("--n_noise_mags", type=int, default=65)
    parser.add_argument("--noise_energy_smooth_weight", type=float, default=0.3)
    parser.add_argument("--noise_mag_smooth_weight", type=float, default=0.3)
    parser.add_argument("--noise_energy_size_weight", type=float, default=0.02)

    parser.add_argument("--amp_smooth_weight", type=float, default=1.0)
    parser.add_argument("--amp_curvature_weight", type=float, default=1.0)
    parser.add_argument("--envelope_weight", type=float, default=3.0)
    parser.add_argument("--log_mag_weight", type=float, default=1.0)

    parser.add_argument("--harmonic_smooth_weight", type=float, default=8.0)
    parser.add_argument("--harmonic_curvature_weight", type=float, default=8.0)

    parser.add_argument("--brightness_smooth_weight", type=float, default=1.0)
    parser.add_argument("--brightness_curvature_weight", type=float, default=1.0)

    parser.add_argument("--residual_size_weight", type=float, default=0.05)

    parser.add_argument("--harmonic_profile_weight", type=float, default=8.0)
    parser.add_argument("--harmonic_profile_log_weight", type=float, default=1.2)
    parser.add_argument("--harmonic_over_weight", type=float, default=0.8)
    parser.add_argument("--harmonic_max_over_weight", type=float, default=0.6)

    parser.add_argument("--time_harmonic_profile_weight", type=float, default=2.0)
    parser.add_argument("--time_harmonic_profile_log_weight", type=float, default=0.25)

    parser.add_argument("--transient_energy_weight", type=float, default=0.05)
    parser.add_argument("--transient_late_weight", type=float, default=0.5)
    parser.add_argument("--transient_smooth_weight", type=float, default=0.5)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")

    target_np = load_audio(
        args.target,
        sr=args.sr,
        max_seconds=args.seconds,
    )

    if args.f0 == "auto":
        f0 = estimate_f0(target_np, args.sr)
    else:
        f0 = float(args.f0)

    print(f"Using f0 = {f0:.2f} Hz")

    if args.max_harmonic_freq == "auto":
        max_harmonic_freq = estimate_auto_harmonic_cutoff(
            target_np,
            args.sr,
            f0,
            db_threshold=args.auto_harmonic_db,
            min_cutoff=args.auto_min_cutoff,
            max_cutoff=args.auto_max_cutoff,
        )
    else:
        max_harmonic_freq = float(args.max_harmonic_freq)

    print(f"Using max_harmonic_freq = {max_harmonic_freq:.2f} Hz")

    target = torch.tensor(target_np, device=device)

    model = DDSPHarmonicSynthesizer(
        sr=args.sr,
        n_samples=len(target_np),
        f0=f0,
        n_harmonics=args.harmonics,
        n_frames=args.n_frames,
        max_harmonic_freq=max_harmonic_freq,
        residual_strength=args.residual_strength,
        brightness_strength=args.brightness_strength,
        amp_smooth_kernel=args.amp_smooth_kernel,
        harmonic_smooth_kernel=args.harmonic_smooth_kernel,
        brightness_smooth_kernel=args.brightness_smooth_kernel,
        use_transient_noise=not args.disable_transient_noise,
        transient_decay=args.transient_decay,
        use_broadband_noise=not args.disable_broadband_noise,
        n_noise_mags=args.n_noise_mags,
    ).to(device)

    print(f"Effective harmonics = {model.n_harmonics}")
    print(f"Highest harmonic = {model.n_harmonics * f0:.2f} Hz")
    print(f"Transient noise enabled = {not args.disable_transient_noise}")
    print(f"Broadband filtered noise enabled = {not args.disable_broadband_noise}")
    print(
        "Model: harmonic + noise synthesis "
        "(envelope, transient noise, brightness decay, continuous filtered noise)"
    )

    target_harmonic_profile_np = estimate_target_harmonic_profile(
        target_np,
        args.sr,
        f0,
        model.n_harmonics,
    )

    target_harmonic_profile = torch.tensor(
        target_harmonic_profile_np,
        dtype=torch.float32,
        device=device,
    )

    target_time_profile_np, target_time_weights_np = (
        estimate_target_harmonic_profile_frames(
            target_np,
            args.sr,
            f0,
            model.n_harmonics,
            args.n_frames,
        )
    )

    target_time_profile = torch.tensor(
        target_time_profile_np,
        dtype=torch.float32,
        device=device,
    )

    target_time_weights = torch.tensor(
        target_time_weights_np,
        dtype=torch.float32,
        device=device,
    )

    target_env = audio_rms_envelope(target, args.n_frames).detach()

    late_mask = torch.linspace(0.0, 1.0, args.n_frames, device=device)
    late_mask = late_mask**2

    print("Target harmonic profile:")
    print(target_harmonic_profile_np)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    losses = []

    early_stopping_enabled = not args.disable_early_stopping
    min_steps = max(1, min(args.early_stop_min_steps, args.steps))
    check_interval = max(1, args.early_stop_check_interval)
    patience_checks = max(1, args.early_stop_patience)
    min_relative_improvement = max(0.0, args.early_stop_min_relative_improvement)

    # best_state tracks the lowest raw loss seen at any step.
    # monitor_best_loss is used only for plateau detection at check intervals.
    best_loss = float("inf")
    best_step = 0
    best_state = None
    monitor_best_loss = float("inf")
    checks_without_improvement = 0
    stopped_early = False
    completed_steps = 0

    if early_stopping_enabled:
        print(
            "Early stopping enabled: "
            f"min_steps={min_steps}, "
            f"check_interval={check_interval}, "
            f"patience={patience_checks}, "
            f"min_relative_improvement={min_relative_improvement:.4f}"
        )
    else:
        print("Early stopping disabled.")

    for step in range(1, args.steps + 1):
        optimizer.zero_grad()

        output, params = model()

        spec_loss = multi_scale_stft_loss(
            output, target, log_mag_weight=args.log_mag_weight
        )

        output_env = audio_rms_envelope(output, args.n_frames)
        envelope_loss = torch.mean(torch.abs(output_env - target_env))

        amp_smooth = smoothness_1d(params["amp_frames"])
        amp_curve = curvature_1d(params["amp_frames"])

        harmonic_smooth = smoothness_2d_time(params["harmonic_distribution_frames"])
        harmonic_curve = curvature_2d_time(params["harmonic_distribution_frames"])

        brightness_smooth = smoothness_1d(params["brightness_frames"])
        brightness_curve = curvature_1d(params["brightness_frames"])

        residual_size = torch.mean(torch.abs(params["harmonic_residual"]))

        mean_harmonic_distribution = torch.mean(
            params["harmonic_distribution_frames"],
            dim=0,
        )

        harmonic_profile_loss = torch.mean(
            torch.abs(mean_harmonic_distribution - target_harmonic_profile)
        )

        harmonic_profile_log_loss = log_harmonic_profile_loss(
            mean_harmonic_distribution,
            target_harmonic_profile,
        )

        harmonic_over_loss = harmonic_over_penalty(
            mean_harmonic_distribution,
            target_harmonic_profile,
        )

        harmonic_max_over_loss = harmonic_max_over_penalty(
            mean_harmonic_distribution,
            target_harmonic_profile,
        )

        time_profile_l1, time_profile_log = time_harmonic_profile_loss(
            params["harmonic_distribution_frames"],
            target_time_profile,
            target_time_weights,
        )

        transient_energy = torch.mean(params["transient_env_frames"])
        transient_late = torch.mean(params["transient_env_frames"] * late_mask)
        transient_smooth = smoothness_1d(params["transient_env_frames"])

        noise_energy_smooth = smoothness_1d(params["noise_energy_frames"])
        noise_mag_smooth = smoothness_2d_time(params["noise_mag_frames"])
        noise_energy_size = torch.mean(params["noise_energy_frames"])

        loss = (
            spec_loss
            + args.envelope_weight * envelope_loss
            + args.amp_smooth_weight * amp_smooth
            + args.amp_curvature_weight * amp_curve
            + args.harmonic_smooth_weight * harmonic_smooth
            + args.harmonic_curvature_weight * harmonic_curve
            + args.brightness_smooth_weight * brightness_smooth
            + args.brightness_curvature_weight * brightness_curve
            + args.residual_size_weight * residual_size
            + args.harmonic_profile_weight * harmonic_profile_loss
            + args.harmonic_profile_log_weight * harmonic_profile_log_loss
            + args.harmonic_over_weight * harmonic_over_loss
            + args.harmonic_max_over_weight * harmonic_max_over_loss
            + args.time_harmonic_profile_weight * time_profile_l1
            + args.time_harmonic_profile_log_weight * time_profile_log
            + args.transient_energy_weight * transient_energy
            + args.transient_late_weight * transient_late
            + args.transient_smooth_weight * transient_smooth
            + args.noise_energy_smooth_weight * noise_energy_smooth
            + args.noise_mag_smooth_weight * noise_mag_smooth
            + args.noise_energy_size_weight * noise_energy_size
        )

        loss.backward()
        optimizer.step()

        current_loss = float(loss.detach().cpu())
        losses.append(current_loss)
        completed_steps = step

        # Keep a copy of the best parameter state, not merely the last state.
        if current_loss < best_loss:
            best_loss = current_loss
            best_step = step
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

        if step == 1 or step % 50 == 0:
            print(
                f"Step {step:04d} | "
                f"Loss: {current_loss:.5f} | "
                f"Spec: {spec_loss.item():.5f} | "
                f"Env: {envelope_loss.item():.5f} | "
                f"TimeProfile: {time_profile_l1.item():.5f}"
            )

        if early_stopping_enabled and step >= min_steps and step % check_interval == 0:
            if monitor_best_loss == float("inf"):
                significant_improvement = True
            else:
                required_loss = monitor_best_loss * (1.0 - min_relative_improvement)
                significant_improvement = current_loss < required_loss

            if significant_improvement:
                monitor_best_loss = current_loss
                checks_without_improvement = 0
                print(
                    f"Early-stop check at step {step}: improved "
                    f"(monitor loss={monitor_best_loss:.5f})."
                )
            else:
                checks_without_improvement += 1
                print(
                    f"Early-stop check at step {step}: no meaningful improvement "
                    f"({checks_without_improvement}/{patience_checks})."
                )

            if checks_without_improvement >= patience_checks:
                stopped_early = True
                print(
                    f"\nEarly stopping at step {step}. "
                    f"Best loss={best_loss:.5f} at step {best_step}."
                )
                break

    # Always render and save the best parameter state found during training.
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best model from step {best_step} (loss={best_loss:.5f}).")

    with torch.no_grad():
        output, params = model()
        output_env = audio_rms_envelope(output, args.n_frames)

    output_np = output.detach().cpu().numpy()
    output_env_np = output_env.detach().cpu().numpy()

    harmonic_frames = params["harmonic_distribution_frames"].detach().cpu().numpy()
    amp_frames = params["amp_frames"].detach().cpu().numpy()
    brightness_frames = params["brightness_frames"].detach().cpu().numpy()
    transient_env_frames = params["transient_env_frames"].detach().cpu().numpy()
    output_gain = float(params["gain"].detach().cpu())
    transient_gain = float(params["transient_gain"].detach().cpu())
    noise_mag_frames = params["noise_mag_frames"].detach().cpu().numpy()
    noise_energy_frames = params["noise_energy_frames"].detach().cpu().numpy()
    noise_gain = float(params["noise_gain"].detach().cpu())

    learned_profile_np = harmonic_frames.mean(axis=0)

    save_wav(out_dir / "output.wav", output_np, args.sr)
    save_wav(out_dir / "target_normalized.wav", target_np, args.sr)

    plot_loss(losses, out_dir)

    plot_spectrogram(
        target_np,
        args.sr,
        "Target Spectrogram",
        out_dir / "target_spectrogram.png",
    )

    plot_spectrogram(
        output_np,
        args.sr,
        "Output Spectrogram",
        out_dir / "output_spectrogram.png",
    )

    plot_mean_harmonics(harmonic_frames, f0, out_dir)
    plot_target_vs_learned_profile(
        target_harmonic_profile_np,
        learned_profile_np,
        f0,
        out_dir,
    )
    plot_harmonic_heatmap(harmonic_frames, f0, out_dir)
    plot_amp_envelope(output_env_np, target_env.detach().cpu().numpy(), out_dir)
    plot_brightness(brightness_frames, out_dir)
    plot_transient(transient_env_frames, out_dir)

    learned_info = {
        "target_file": str(args.target),
        "sample_rate": args.sr,
        "f0_hz": f0,
        "max_harmonic_freq": max_harmonic_freq,
        "auto_harmonic_db": args.auto_harmonic_db,
        "effective_harmonics": model.n_harmonics,
        "highest_harmonic_hz": model.n_harmonics * f0,
        "transient_noise_enabled": not args.disable_transient_noise,
        "output_gain": output_gain,
        "transient_gain": transient_gain,
        "broadband_noise_enabled": not args.disable_broadband_noise,
        "noise_gain": noise_gain,
        "noise_mag_frames": noise_mag_frames.tolist(),
        "noise_energy_frames": noise_energy_frames.tolist(),
        "final_loss": best_loss,
        "last_observed_loss": losses[-1],
        "best_loss": best_loss,
        "best_step": best_step,
        "completed_steps": completed_steps,
        "requested_max_steps": args.steps,
        "early_stopping_enabled": early_stopping_enabled,
        "stopped_early": stopped_early,
        "early_stop_min_steps": min_steps,
        "early_stop_check_interval": check_interval,
        "early_stop_patience": patience_checks,
        "early_stop_min_relative_improvement": min_relative_improvement,
        "target_harmonic_profile": target_harmonic_profile_np.tolist(),
        "learned_harmonic_profile": learned_profile_np.tolist(),
        "harmonic_distribution_frames": harmonic_frames.tolist(),
        "amp_frames": amp_frames.tolist(),
        "brightness_frames": brightness_frames.tolist(),
        "transient_env_frames": transient_env_frames.tolist(),
        "description": "Harmonic + noise synthesis with envelope matching, transient noise, brightness decay, and time-varying harmonic profile loss.",
    }

    with open(out_dir / "learned_params.json", "w") as f:
        json.dump(learned_info, f, indent=2)

    print("\nDone.")
    print(
        f"Training steps: {completed_steps}/{args.steps} | "
        f"best step: {best_step} | best loss: {best_loss:.5f}"
    )
    print(f"Saved: {out_dir / 'output.wav'}")
    print(f"Saved: {out_dir / 'target_normalized.wav'}")
    print(f"Saved: {out_dir / 'loss_curve.png'}")
    print(f"Saved: {out_dir / 'target_spectrogram.png'}")
    print(f"Saved: {out_dir / 'output_spectrogram.png'}")
    print(f"Saved: {out_dir / 'mean_harmonic_distribution.png'}")
    print(f"Saved: {out_dir / 'target_vs_learned_harmonic_profile.png'}")
    print(f"Saved: {out_dir / 'harmonic_distribution_heatmap.png'}")
    print(f"Saved: {out_dir / 'amplitude_envelope.png'}")
    print(f"Saved: {out_dir / 'brightness_damping.png'}")
    print(f"Saved: {out_dir / 'transient_noise_envelope.png'}")
    print(f"Saved: {out_dir / 'learned_params.json'}")


if __name__ == "__main__":
    main()
