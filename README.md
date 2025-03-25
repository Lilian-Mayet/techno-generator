# 🎧 Techno Music Generation with Variational Autoencoder (VAE)

This project uses Machine Learning to generate original techno music segments using a Variational Autoencoder (VAE). By training on short spectrogram excerpts from techno songs, the model learns to encode and decode spectrogram representations, enabling it to produce coherent and original techno music.

---

## 🎯 Project Goal

The primary goal is to automatically generate musically coherent techno tracks using deep learning techniques. Specifically, the project aims to:

- Learn meaningful latent representations of techno music using a VAE.
- Generate new techno music segments by sampling and interpolating in the learned latent space.
- Create longer, coherent techno tracks by smoothly transitioning between generated segments.

---

## ⚙️ Process Overview

### 1. **Data Preparation**
- Collect short audio excerpts (around 10 seconds each) from existing techno tracks.
- Convert audio files into Mel-spectrograms for effective representation.

### 2. **Training the Variational Autoencoder (VAE)**
- Train a convolutional encoder-decoder neural network to reconstruct Mel-spectrograms.
- Optimize model parameters using reconstruction loss and KL divergence.
- Dynamically adjust the learning rate during training for optimal performance.

### 3. **Generating New Music Segments**
- Randomly sample latent vectors from the learned latent space.
- Decode these vectors into Mel-spectrograms.
- Convert generated spectrograms back into audio using the Griffin-Lim algorithm.

### 4. **Creating Longer Techno Tracks**
- Begin with a logical, existing techno segment from the dataset.
- Generate subsequent segments by smoothly interpolating between latent vectors.
- Combine segments using audio crossfades to achieve smooth transitions.

---

## 🎶 Project Structure
```
├── spectrogram_dataset.npy         # Training dataset (excluded from GitHub)
├── vae_model.py                    # Main script for training the VAE
├── generate_song.py                # Generate short techno samples
├── generate_long_song.py           # Generate longer techno tracks
├── saved_models/                   # Directory for saved models
│   └── vae_best.keras              # Best trained VAE model
└── README.md                       # Project description and guide
```

---

## 🚀 How to Run

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Training the Model

```bash
python vae_model.py
```

### Generate Short Samples

```bash
python generate_song.py
```

### Generate Longer, Logical Techno Tracks

```bash
python generate_long_song.py
```

---

## 📌 Notes
- The original dataset (`spectrogram_dataset.npy`) and generated audio files are excluded from this repository to maintain a small repository size.
- Ensure your dataset is correctly prepared before training.

---

✨ **Enjoy your AI-generated techno beats!** 🎧

