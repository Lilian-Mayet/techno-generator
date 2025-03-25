import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Flatten, Dense, Reshape, Lambda, Cropping2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau,LearningRateScheduler

import os
import matplotlib.pyplot as plt

# Parameters
input_shape = (128, 431, 1)  # Adjust based on your spectrogram shape
latent_dim = 64
epochs = 25
batch_size = 16

# Custom sampling function
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.keras.backend.random_normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Encoder definition (optimized)
def build_encoder(input_shape, latent_dim):
    inputs = Input(shape=input_shape, name='encoder_input')
    x = Conv2D(32, 3, activation='relu', strides=2, padding='same')(inputs)
    x = Conv2D(64, 3, activation='relu', strides=2, padding='same')(x)
    x = Conv2D(128, 3, activation='relu', strides=2, padding='same')(x)  # Extra conv layer
    shape_before_flatten = x.shape[1:]
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)  # Smaller Dense layer

    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    z = Lambda(sampling, name='z')([z_mean, z_log_var])

    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    return encoder, shape_before_flatten

# Decoder definition with cropping fix
def build_decoder(shape_before_flatten, latent_dim):
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(np.prod(shape_before_flatten), activation='relu')(latent_inputs)
    x = Reshape(shape_before_flatten)(x)

    x = Conv2DTranspose(128, 3, activation='relu', strides=2, padding='same')(x)
    x = Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')(x)
    x = Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same')(x)
    x = Conv2DTranspose(1, 3, activation='sigmoid', padding='same')(x)

    # Crop to match input exactly
    decoder_outputs = Cropping2D(cropping=((0,0),(0,1)))(x)

    decoder = Model(latent_inputs, decoder_outputs, name='decoder')
    return decoder

# Custom VAE Loss Layer
class VAELossLayer(tf.keras.layers.Layer):
    def __init__(self, input_shape, **kwargs):
        super(VAELossLayer, self).__init__(**kwargs)
        self.input_shape_ = input_shape

    def call(self, inputs):
        original, reconstructed, z_mean, z_log_var = inputs
        reconstruction_loss = tf.reduce_mean(
            tf.square(original - reconstructed), axis=[1,2,3]
        ) * self.input_shape_[0] * self.input_shape_[1]

        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1
        )

        vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
        self.add_loss(vae_loss)

        return reconstructed

# Compile VAE
def compile_vae(encoder, decoder, input_shape):
    inputs = encoder.input
    z_mean, z_log_var, z = encoder(inputs)
    reconstructed = decoder(z)

    outputs = VAELossLayer(input_shape)([inputs, reconstructed, z_mean, z_log_var])
    vae = Model(inputs, outputs, name='vae')
    vae.compile(optimizer=Adam(learning_rate=0.0005))
    return vae

# Load data
def load_data(filepath):
    spectrograms = np.load(filepath)[..., np.newaxis]
    spectrograms = (spectrograms - np.min(spectrograms)) / (np.max(spectrograms) - np.min(spectrograms))
    return spectrograms




def lr_schedule(epoch, lr):
    initial_lr = 0.00006
    decay_rate = 0.95  # Decay learning rate by 5% each epoch
    new_lr = initial_lr * decay_rate ** epoch
    return max(new_lr, 1e-6)  # Minimum LR of 1e-6

def train_vae(vae, data, epochs, batch_size, model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    checkpoint = ModelCheckpoint(
        model_path, save_best_only=True, monitor='val_loss', verbose=1
    )

    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)

    history = vae.fit(
        data, data,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[checkpoint, lr_scheduler]
    )
    np.save('saved_models/training_history.npy', history.history)
    return history
# Main execution
def main():
    data = load_data('spectrogram_dataset.npy')

    encoder, shape_before_flatten = build_encoder(input_shape, latent_dim)
    decoder = build_decoder(shape_before_flatten, latent_dim)

    # Load existing model if available
    model_path = 'saved_models/vae_best.keras'
    if os.path.exists(model_path):
        print("Loading existing model...")
        vae = load_model(model_path, custom_objects={'VAELossLayer': VAELossLayer, 'sampling': sampling})
    else:
        vae = compile_vae(encoder, decoder, input_shape)

    history = train_vae(vae, data, epochs, batch_size, model_path)
     
    history_dict = history.history

    plt.figure(figsize=(10,6))
    plt.plot(history_dict['loss'], label='Training Loss')
    plt.plot(history_dict['val_loss'], label='Validation Loss')

    plt.title('Training and Validation Loss', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)

    plt.savefig('saved_models/training_history.png')  # save plot image
    plt.show()

if __name__ == "__main__":
    main()
