import tensorflow as tf
from tensorflow.keras.layers import Layer

class VAELossLayer(Layer):
    def __init__(self, input_shape, **kwargs):
        super(VAELossLayer, self).__init__(**kwargs)
        self.input_shape_ = input_shape

    def call(self, inputs):
        original, reconstructed, z_mean, z_log_var = inputs

        # Compute reconstruction loss directly (element-wise difference)
        reconstruction_loss = tf.reduce_mean(
            tf.square(original - reconstructed), axis=[1, 2, 3]
        )
        reconstruction_loss *= self.input_shape_[0] * self.input_shape_[1]

        # Compute KL divergence
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = -0.5 * tf.reduce_mean(kl_loss, axis=-1)

        # Total loss
        vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)

        self.add_loss(vae_loss)

        # Simply return reconstructed for output
        return reconstructed
