import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import librosa
import soundfile as sf
import random
from vae_model import VAELossLayer, sampling
from pydub import AudioSegment

latent_dim = 64
model_path = 'saved_models/vae_best.keras'

# --- Custom Layers ---
class Sampling(Layer):
    def __init__(self, latent_dim, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim

    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.keras.backend.random_normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Load decoder
vae = load_model(model_path, custom_objects={
    'VAELossLayer': VAELossLayer,
    'sampling': Sampling(latent_dim=latent_dim)
})
decoder = vae.get_layer('decoder')
encoder = vae.get_layer('encoder')

# --- Interpolation between latent vectors ---
def interpolate_vectors(v1, v2, steps=4):
    vectors = []
    for alpha in np.linspace(0, 1, steps):
        vector = (1 - alpha) * v1 + alpha * v2
        vectors.append(vector)
    return vectors

# --- Generate spectrogram from latent vector ---
def generate_spectrogram(decoder, latent_vector):
    generated_spec = decoder.predict(latent_vector[np.newaxis, :])[0].squeeze()
    return generated_spec

# --- Convert spectrogram to audio ---
def spectrogram_to_audio(mel_spectrogram, sr=22050, hop_length=512, n_iter=64):
    mel_spectrogram_db = (mel_spectrogram * 80.0) - 80.0
    mel_spectrogram_power = librosa.db_to_power(mel_spectrogram_db)
    audio = librosa.feature.inverse.mel_to_audio(
        mel_spectrogram_power,
        sr=sr,
        hop_length=hop_length,
        n_iter=n_iter
    )
    audio = audio / np.max(np.abs(audio))
    return audio

# --- Load existing spectrogram dataset and encode to latent ---
def load_and_encode_random_existing_segment(encoder, dataset_path):
    data = np.load(dataset_path)[..., np.newaxis]
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    random_index = np.random.randint(0, len(data))
    random_spectrogram = data[random_index:random_index+1]

    # Encode to latent space
    z_mean, z_log_var, z = encoder.predict(random_spectrogram)
    return z.squeeze()



def crossfade_segments(audio_segments, sr, crossfade_duration_ms=1500):
    # Convert numpy arrays to pydub segments
    segments_pydub = [AudioSegment(
        (segment * 32767).astype(np.int16).tobytes(),
        frame_rate=sr,
        sample_width=2,
        channels=1
    ) for segment in audio_segments]

    # Apply crossfade sequentially
    final_audio = segments_pydub[0]
    for segment in segments_pydub[1:]:
        final_audio = final_audio.append(segment, crossfade=crossfade_duration_ms)

    # Convert back to numpy array
    final_audio_np = np.array(final_audio.get_array_of_samples()) / 32767.0
    return final_audio_np


# --- Generate a long logical techno song ---
def generate_long_song(decoder, encoder, latent_dim, segments=8, dataset_path='spectrogram_dataset.npy'):
    sr = 22050
    hop_length = 512
    song_audio = np.array([])

    # First segment is an existing logical techno segment
    prev_vector = load_and_encode_random_existing_segment(encoder, dataset_path)

    for i in range(segments):
        next_vector = np.random.normal(size=(latent_dim,))
        interpolated_vectors = interpolate_vectors(prev_vector, next_vector, steps=3)

        for vec in interpolated_vectors:
            spec = generate_spectrogram(decoder, vec)
            audio_segment = spectrogram_to_audio(spec, sr, hop_length)
            song_audio = np.concatenate((song_audio, audio_segment))

        prev_vector = next_vector

    return song_audio, sr

# --- Main execution ---
def main():
    segments = random.randint(2, 2)
    print(f"Generating techno song with {segments} segments...")

    audio, sr = generate_long_song(
        decoder, encoder, latent_dim, segments=segments, dataset_path='spectrogram_dataset.npy'
    )

    sf.write('logical_long_techno_song.wav', audio, sr)
    print("Logical long techno song generated as 'logical_long_techno_song.wav'.")

if __name__ == "__main__":
    main()
