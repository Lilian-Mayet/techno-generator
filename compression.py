import numpy as np
import matplotlib.pyplot as plt

# Load spectrogram dataset (in dB)
data = np.load('spectrogram_dataset.npy')

# Parameters
num_bands = 4
points_per_band = 150

# Function to process a single spectrogram into frequency bands and select important points
def process_spectrogram(spectrogram, num_bands=4, points_per_band=100):
    freq_bins, time_steps = spectrogram.shape
    band_size = freq_bins // num_bands
    print(freq_bins)

    selected_points = []

    for band in range(num_bands):
        start = band * band_size
        end = (band + 1) * band_size if band < num_bands - 1 else freq_bins
        freq_band = spectrogram[start:end, :]

        # Find indices of the most significant points
        flat_indices = np.argpartition(freq_band.flatten(), -points_per_band)[-points_per_band:]

        # Convert flat indices to 2D coordinates (frequency, time)
        freq_coords, time_coords = np.unravel_index(flat_indices, freq_band.shape)

        # Adjust coordinates to actual spectrogram position
        freq_coords += start

        # Save points as (frequency, time, amplitude)
        band_points = [(f, t, spectrogram[f, t]) for f, t in zip(freq_coords, time_coords)]
        selected_points.extend(band_points)

    return selected_points

# Function to process the entire dataset
def process_all_spectrograms(dataset, num_bands=4, points_per_band=100):
    return [process_spectrogram(spectrogram, num_bands, points_per_band) for spectrogram in dataset]

def plot_random_spectrogram(dataset, num_bands=4, points_per_band=100):
    idx = np.random.randint(len(dataset))
    spectrogram = dataset[idx]
    points = process_spectrogram(spectrogram, num_bands, points_per_band)

    plt.figure(figsize=(12, 6))
    plt.imshow(spectrogram, aspect='auto', cmap='magma', origin='lower')
    plt.colorbar(label='Amplitude (dB)')

    freq_coords = [p[0] for p in points]
    time_coords = [p[1] for p in points]

    plt.scatter(time_coords, freq_coords, marker='x', color='cyan', s=15)

    # Plot lines separating frequency bands
    band_size = spectrogram.shape[0] // num_bands
    for band in range(1, num_bands):
        plt.axhline(y=band * band_size, color='white', linestyle='--', linewidth=1)

    plt.title('Significant Points and Frequency Bands on Random Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.show()

# Process and save compressed spectrograms
#compressed_spectrograms = process_all_spectrograms(data, num_bands, points_per_band)
#np.save('reduced_spectrogram_bands.npy', compressed_spectrograms)

# Display data sizes
#print(f"Compression complete. Original size: {data.nbytes/1e6:.2f} MB")
#print(f"Compressed size: {np.array(compressed_spectrograms).nbytes/1e6:.2f} MB")

# Plot points on a random spectrogram
plot_random_spectrogram(data, num_bands, points_per_band)