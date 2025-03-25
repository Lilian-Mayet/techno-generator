import os
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm

# Directory containing audio files
AUDIO_DIR = 'song_segmented'

# Parameters for spectrogram
SR = 22050           # Sample rate (adjust as needed)
DURATION = 10        # Duration of audio extracts (seconds)
N_MELS = 128         # Number of Mel frequency bins
HOP_LENGTH = 512     # Hop length for spectrogram (overlap)

def audio_to_mel_spectrogram(audio_path, sr=SR, duration=DURATION, n_mels=N_MELS, hop_length=HOP_LENGTH):
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=sr, duration=duration)
    
    # Ensure audio length is consistent
    expected_len = int(sr * duration)
    if len(y) < expected_len:
        # Pad audio if too short
        y = np.pad(y, (0, expected_len - len(y)))
    else:
        y = y[:expected_len]
    
    # Compute Mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    
    # Convert to dB scale (log-scale)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    return S_dB

def process_directory(audio_dir):
    spectrograms = []
    filenames = []

    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav') or f.endswith('.mp3')]

    for audio_file in tqdm(audio_files, desc='Processing audio files'):
        audio_path = os.path.join(audio_dir, audio_file)
        mel_spec = audio_to_mel_spectrogram(audio_path)
        spectrograms.append(mel_spec)
        filenames.append(audio_file)

    spectrograms = np.array(spectrograms)
    return spectrograms, filenames

if __name__ == "__main__":
    spectrogram_dataset, filenames = process_directory(AUDIO_DIR)
    
    # Save spectrograms as numpy arrays for your autoencoder
    np.save('spectrogram_dataset.npy', spectrogram_dataset)
    
    print(f"Spectrogram dataset shape: {spectrogram_dataset.shape}")
    print(f"Saved spectrogram_dataset.npy successfully.")
