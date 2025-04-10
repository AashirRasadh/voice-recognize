import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import numpy as np
import matplotlib.pyplot as plt
import os

# === 1. Record Audio ===
duration = 5  # seconds
sample_rate = 22050  # Hz

print("Recording...")
audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
sd.wait()
print("Recording finished.")

# Save to a temporary file
output_path = "mic_input.wav"
write(output_path, sample_rate, audio)

# === 2. Load and Process Audio ===
y, sr = librosa.load(output_path, sr=None)

f0, voiced_flag, voiced_probs = librosa.pyin(
    y,
    fmin=librosa.note_to_hz('C2'),
    fmax=librosa.note_to_hz('C7')
)

f0_clean = f0[~np.isnan(f0)]

if len(f0_clean) == 0:
    print("No voiced speech detected.")
else:
    avg_pitch = np.mean(f0_clean)
    print(f"Average Fundamental Frequency: {avg_pitch:.2f} Hz")

    if avg_pitch < 165:
        print("Detected Voice: Male")
    else:
        print("Detected Voice: Female")

    # === Optional: Plot pitch curve ===
    plt.figure(figsize=(10, 4))
    times = librosa.times_like(f0)
    plt.plot(times, f0, label='F0', color='purple')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Fundamental Frequency Estimation')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Optional: remove temp audio file
if os.path.exists(output_path):
    os.remove(output_path)
