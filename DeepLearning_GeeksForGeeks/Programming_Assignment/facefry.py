import urllib.request
import os
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf


BATCH_SIZE = 128
LATENT_DIM = 20
ORIGINAL_IMAGE_SHAPE = (28, 20)
ORIGINAL_IMAGE_DIM = np.prod(ORIGINAL_IMAGE_SHAPE)
EPOCHS = 200
LEARNING_RATE = 1e-3
DATASET_URL = "https://cs.nyu.edu/home/people/in_memoriam/roweis/data/frey_rawface.mat"
DATASET_FILENAME = "frey_rawface.mat"

if not os.path.exists(DATASET_FILENAME):
    urllib.request.urlretrieve(DATASET_URL, DATASET_FILENAME)

data = scipy.io.loadmat(
    DATASET_FILENAME, squeeze_me=True, struct_as_record=False)
faces = data['ff'].T.astype('float32')
faces /= 255.0
faces_flat = faces.reshape(-1, ORIGINAL_IMAGE_DIM)

dataset = tf.data.Dataset.from_tensor_slices(faces_flat)
dataset = dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)


class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


inputs = keras.Input(shape=(ORIGINAL_IMAGE_DIM,))
x = layers.Dense(256, activation="relu")(inputs)
z_mean = layers.Dense(LATENT_DIM, name="z_mean")(x)
z_log_var = layers.Dense(LATENT_DIM, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")

inputs = keras.Input(shape=(LATENT_DIM,))
x = layers.Dense(256, activation="relu")(inputs)
outputs = layers.Dense(ORIGINAL_IMAGE_DIM, activation="sigmoid")(x)
decoder = keras.Model(inputs, outputs, name="decoder")


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(data, reconstruction)
            )
            reconstruction_loss *= ORIGINAL_IMAGE_DIM
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE))
vae.fit(dataset, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)

original = next(iter(dataset))[:10]
_, _, z = vae.encoder.predict(original)
reconstructed = vae.decoder.predict(z)

original = original.numpy().reshape(-1,
                                    ORIGINAL_IMAGE_SHAPE[0], ORIGINAL_IMAGE_SHAPE[1])
reconstructed = reconstructed.reshape(-1,
                                      ORIGINAL_IMAGE_SHAPE[0], ORIGINAL_IMAGE_SHAPE[1])

plt.figure(figsize=(20, 4))
for i in range(10):
    ax = plt.subplot(2, 10, i + 1)
    plt.imshow(original[i], cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if i == 5:
        ax.set_title("Original Images")

    ax = plt.subplot(2, 10, i + 11)
    plt.imshow(reconstructed[i], cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if i == 5:
        ax.set_title("Reconstructed Images")
plt.show()

figure = np.zeros((28, 20 * 15))
grid = np.random.normal(size=(15, LATENT_DIM))
reconstructions = decoder.predict(grid)

for i in range(15):
    face = reconstructions[i].reshape((28, 20))
    figure[:, i * 20: (i + 1) * 20] = face

plt.figure(figsize=(15, 5))
plt.imshow(figure, cmap="gray")
plt.title("Generated Faces from Random Latent Vectors")
plt.axis('off')
plt.show()

sample = faces[np.random.choice(len(faces))].reshape(1, -1)
z_mean, _, _ = encoder.predict(sample)

fig, axes = plt.subplots(1, 10, figsize=(15, 3))

for i, val in enumerate(np.linspace(-2.0, 2.0, 10)):
    latent = np.copy(z_mean)
    latent[0, 4] = val
    decoded = decoder.predict(latent)
    ax = axes[i]
    ax.imshow(decoded.reshape(28, 20), cmap='gray')
    ax.set_title(f'val={val:.1f}', fontsize=8)
    ax.axis('off')

fig.suptitle(f'Traversal of Latent Dimension 4')
