import librosa
import numpy as np
import matplotlib.pyplot as plt

# Set the path to your .wav file
file_path = 'D:/Github Document/voice-recognize/audio_files/test1.wav'

# Load the audio
y, sr = librosa.load(file_path)

# Estimate the fundamental frequency (F0)
f0, voiced_flag, voiced_probs = librosa.pyin(
    y,
    fmin=librosa.note_to_hz('C2'),
    fmax=librosa.note_to_hz('C7')
)

# Remove unvoiced sections
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

    # Optional: plot the pitch curve
    plt.figure(figsize=(10, 4))
    times = librosa.times_like(f0)
    plt.plot(times, f0, label='F0', color='purple')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Fundamental Frequency Estimation')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
