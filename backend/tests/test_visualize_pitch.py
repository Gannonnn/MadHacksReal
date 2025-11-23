import librosa
import numpy as np
import torch
from pathlib import Path
import sys
from librosa.core import hz_to_mel, mel_frequencies
from loguru import logger
from matplotlib import pyplot as plt

# Ensure backend directory (which contains fish_diffusion) is on sys.path
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from fish_diffusion.modules.pitch_extractors import (
    #CrepePitchExtractor,
    #DioPitchExtractor,
    #HarvestPitchExtractor,
    ParselMouthPitchExtractor,
    #RMVPitchExtractor,
)
from fish_diffusion.utils.audio import get_mel_from_audio

device = "cpu"

f_min = 40
f_max = 16000
n_mels = 128

min_mel = hz_to_mel(f_min)
max_mel = hz_to_mel(f_max)
f_to_mel = lambda x: (hz_to_mel(x) - min_mel) / (max_mel - min_mel) * n_mels
mel_freqs = mel_frequencies(n_mels=n_mels, fmin=f_min, fmax=f_max)

# Get the path relative to this test file
test_dir = Path(__file__).parent
dataset_path = test_dir.parent / "dataset" / "67.mp3"

audio, sr = librosa.load(
    str(dataset_path),
    sr=44100,
    mono=True,
)

# beat tracking
def get_beat_tempo():
    audio, sr = librosa.load(
        str(dataset_path),
        sr=44100,
        mono=True,
    )
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    return tempo

# Store original audio length before converting to tensor
audio_length = len(audio)
duration = audio_length / sr  # Duration in seconds

audio = torch.from_numpy(audio).unsqueeze(0).to(device)

mel = (
    get_mel_from_audio(audio, sr, f_min=f_min, f_max=f_max, n_mels=n_mels).cpu().numpy()
)
logger.info(f"Got mel spectrogram with shape {mel.shape}")

# Calculate time arrays - mel spectrogram uses hop_length=512 by default
hop_length = 512
n_frames = mel.shape[-1]
# Time for each frame: frame_index * hop_length / sr
time_frames = np.arange(n_frames) * hop_length / sr

extractors = {
    #"RMVPE": RMVPitchExtractor,
    #"Crepe": CrepePitchExtractor,
    "ParselMouth": ParselMouthPitchExtractor,
    #"Harvest": HarvestPitchExtractor,
    #"Dio": DioPitchExtractor,
}

images_dir = test_dir.parent / "images"
images_dir.mkdir(exist_ok=True)  # Create folder if it doesn't exist

fig, axs = plt.subplots(len(extractors), 1, figsize=(10, len(extractors) * 3))
fig.suptitle("Pitch/Mel spectrogram")

if len(extractors) == 1:
    axs = [axs]

# Store pitch data for reuse
pitch_data = {}

for idx, (name, extractor) in enumerate(extractors.items()):
    extra_kwargs = {
        "keep_zeros": True,
    }

    pitch_extractor = extractor(f0_min=40.0, f0_max=1600, **extra_kwargs).to(device)
    f0_hz = pitch_extractor(audio, sr, pad_to=mel.shape[-1]).cpu().numpy()
    f0_hz[f0_hz <= 0] = float("nan")
    np.save(images_dir / f"{name.lower()}_pitch_hz.npy", f0_hz)
    f0 = f_to_mel(f0_hz.copy())
    logger.info(f"Got {name} pitch with shape {f0.shape}")
    pitch_data[name] = f0  # Store for unfiltered plot

    # Find the maximum pitch value (excluding NaN) for y-axis scaling
    f0_max_val = np.nanmax(f0)
    # Add 10% padding above max pitch, but don't exceed n_mels
    y_max = min(f0_max_val * 1.1, n_mels) if not np.isnan(f0_max_val) else n_mels

    ax = axs[idx]
    ax.set_title(name)
    # Set extent to show time range on x-axis: [left, right, bottom, top]
    ax.imshow(mel, aspect="auto", origin="lower", extent=[0, duration, 0, n_mels])
    # Plot pitch with time on x-axis
    ax.plot(time_frames, f0, label=name, color="red")
    # Set y-axis limits with 10% padding above max pitch
    ax.set_ylim(0, y_max)
    # Generate y-ticks within the visible range
    # Use smaller spacing (every 3 mel bins) to ensure multiple ticks are visible
    y_tick_positions = np.arange(0, y_max + 1, 2.7)
    y_tick_positions = y_tick_positions[y_tick_positions <= y_max]
    # Convert mel bin positions to Hz values
    y_tick_labels = []
    valid_positions = []
    for pos in y_tick_positions:
        if int(pos) < len(mel_freqs):
            y_tick_labels.append(str(int(round(mel_freqs[int(pos)]))))
            valid_positions.append(pos)
    
    # Set ticks and labels
    ax.set_yticks(valid_positions)
    ax.set_yticklabels(y_tick_labels)
    # Ensure y-axis is visible on the left
    ax.yaxis.set_ticks_position('left')
    ax.yaxis.set_label_position('left')
    # Make sure ticks are visible
    ax.tick_params(axis='y', which='major', labelsize=10, labelcolor='black')
    ax.set_ylabel("Frequency (Hz)", fontsize=12)
    ax.set_xlabel("Time (seconds)")
    ax.legend()

# Adjust layout to ensure labels are visible
plt.tight_layout()

# Save as PNG
output_path = images_dir / "pitch.png"
plt.savefig(str(output_path))
logger.info(f"Saved pitch visualization to {output_path}")

# Save figure as numpy array
fig = plt.gcf()
# Render figure to a buffer and convert to numpy array
fig.canvas.draw()
buf = fig.canvas.buffer_rgba()
fig_array = np.asarray(buf)
numpy_output_path = images_dir / "pitch.npy"
np.save(str(numpy_output_path), fig_array)
logger.info(f"Saved figure as numpy array to {numpy_output_path}")

# ============================================================================
# Create unfiltered pitch visualization
# ============================================================================
fig_unfiltered, axs_unfiltered = plt.subplots(len(extractors), 1, figsize=(10, len(extractors) * 3))
fig_unfiltered.suptitle("Pitch/Mel spectrogram (Unfiltered)")

if len(extractors) == 1:
    axs_unfiltered = [axs_unfiltered]

for idx, (name, extractor) in enumerate(extractors.items()):
    f0 = pitch_data[name]  # Use stored pitch data
    
    ax = axs_unfiltered[idx]
    ax.set_title(name)
    # Use frame indices for x-axis (default fish-diffusion behavior)
    ax.imshow(mel, aspect="auto", origin="lower", extent=[0, n_frames, 0, n_mels])
    # Plot pitch with frame indices on x-axis
    frame_indices = np.arange(n_frames)
    ax.plot(frame_indices, f0, label=name, color="red")
    # Full y-axis range (default fish-diffusion behavior)
    ax.set_ylim(0, n_mels)
    # Set y-ticks and labels - use standard spacing every 10 mel bins
    y_tick_positions = np.arange(0, n_mels + 1, 10)
    y_tick_labels = [str(int(round(mel_freqs[int(pos)]))) for pos in y_tick_positions if int(pos) < len(mel_freqs)]
    min_len = min(len(y_tick_positions), len(y_tick_labels))
    ax.set_yticks(y_tick_positions[:min_len])
    ax.set_yticklabels(y_tick_labels[:min_len])
    ax.yaxis.set_ticks_position('left')
    ax.yaxis.set_label_position('left')
    ax.tick_params(axis='y', which='major', labelsize=10)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Frame")
    ax.legend()

plt.tight_layout()

# Save unfiltered version
unfiltered_output_path = images_dir / "unfiltered_pitch.png"
plt.savefig(str(unfiltered_output_path))
logger.info(f"Saved unfiltered pitch visualization to {unfiltered_output_path}")
# ============================================================================

plt.show()

