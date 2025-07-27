# Install required packages if needed
import urllib.request
import os
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf

BATCH_SIZE = 128
LATENT_DIM = 20  # Size of the latent space (embedding vector)
ORIGINAL_IMAGE_SHAPE = (28, 20)
ORIGINAL_IMAGE_DIM = np.prod(ORIGINAL_IMAGE_SHAPE)
EPOCHS = 200
LEARNING_RATE = 1e-3
DATASET_URL = "https://cs.nyu.edu/home/people/in_memoriam/roweis/data/frey_rawface.mat"
DATASET_FILENAME = "./data/input/q2/frey_rawface.mat"

os.makedirs(os.path.dirname(DATASET_FILENAME), exist_ok=True)
os.makedirs("./data/output/q2", exist_ok=True)
if not os.path.exists(DATASET_FILENAME):
    print("Downloading Frey Face dataset...")
    urllib.request.urlretrieve(DATASET_URL, DATASET_FILENAME)
    print("Download complete.")

mat_data = scipy.io.loadmat(
    DATASET_FILENAME, squeeze_me=True, struct_as_record=False)
frey_faces = mat_data['ff'].T.astype('float32')

frey_faces /= 255.0
frey_faces_flat = frey_faces.reshape(-1, ORIGINAL_IMAGE_DIM)

dataset = tf.data.Dataset.from_tensor_slices(frey_faces_flat)
dataset = dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)
print(f"Dataset shape: {frey_faces_flat.shape}")

"""#### 4. VAE Model Definition"""

# (a) Sampling Layer for the Reparameterization Trick


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding an image."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# (b) Encoder Network
h_dim = 256
encoder_inputs = keras.Input(shape=(ORIGINAL_IMAGE_DIM,))
x = layers.Dense(h_dim, activation="relu")(encoder_inputs)
z_mean = layers.Dense(LATENT_DIM, name="z_mean")(x)
z_log_var = layers.Dense(LATENT_DIM, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

# (c) Decoder Network
latent_inputs = keras.Input(shape=(LATENT_DIM,))
x = layers.Dense(h_dim, activation="relu")(latent_inputs)
decoder_outputs = layers.Dense(ORIGINAL_IMAGE_DIM, activation="sigmoid")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

# (d) VAE Model connecting Encoder and Decoder


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


"""### 5. Training"""

print("Starting training...")
vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE))
vae.fit(dataset, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)
print("Training finished.")

"""### 6. Visualization of Reconstructed Images"""

print("\nDisplaying original vs. reconstructed images...")

# Get the first 10 images from a batch
original_images_flat = next(iter(dataset))[:10]

# Get the reconstructed images from the VAE
_, _, z = vae.encoder.predict(original_images_flat)
reconstructed_images_flat = vae.decoder.predict(z)

# Reshape the flat images back to their original 28x20 shape for display
original_images = original_images_flat.numpy(
).reshape(-1, ORIGINAL_IMAGE_SHAPE[0], ORIGINAL_IMAGE_SHAPE[1])
reconstructed_images = reconstructed_images_flat.reshape(
    -1, ORIGINAL_IMAGE_SHAPE[0], ORIGINAL_IMAGE_SHAPE[1])

# Create a plot to display the images
n = 10  # Number of images to display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original images on the top row
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(original_images[i], cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if i == n // 2:
        ax.set_title("Original Images")

    # Display reconstructed images on the bottom row
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(reconstructed_images[i], cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if i == n // 2:
        ax.set_title("Reconstructed Images")
plt.tight_layout()
plt.savefig("./data/output/q2/reconstructed_images.png")
# plt.show()
plt.close()

"""### 4. Sample from the Latent Space to Generate New Faces"""

# This demonstrates that the VAE can generate novel faces.
n = 15  # Number of faces to generate
digit_size = (28, 20)
figure = np.zeros((digit_size[0], digit_size[1] * n))

# Sample points from the standard normal distribution (the prior)
grid_x = np.random.normal(size=(n, LATENT_DIM))

reconstructions = decoder.predict(grid_x)

for i in range(n):
    face = reconstructions[i].reshape(digit_size)
    figure[:, i * digit_size[1]: (i + 1) * digit_size[1]] = face

plt.figure(figsize=(15, 5))
plt.imshow(figure, cmap="gray")
plt.title("Generated Faces from Random Latent Vectors")
plt.axis('off')
plt.savefig("./data/output/q2/generated_faces.png")
# plt.show()
plt.close()

"""### 5. Varying Latent Variables to Show Learned Features"""

# This visualization traverses one latent dimension while keeping others constant
# to see what visual feature that dimension controls.

# Get the latent representation of a sample image
sample_image = frey_faces[np.random.choice(len(frey_faces))].reshape(1, -1)
z_mean, _, _ = encoder.predict(sample_image)

# Select a dimension to vary
latent_dim_to_vary = 4  # You can change this from 0 to 19
n_steps = 10
min_val, max_val = -2.0, 2.0  # Range of values for the traversal

# Create the figure to display results
fig, axes = plt.subplots(1, n_steps, figsize=(15, 3))

for i, val in enumerate(np.linspace(min_val, max_val, n_steps)):
    # Create a copy of the mean latent vector
    latent_vector = np.copy(z_mean)
    # Modify the chosen dimension
    latent_vector[0, latent_dim_to_vary] = val

    # Decode the modified latent vector
    decoded_image = decoder.predict(latent_vector)

    # Reshape and display the image
    ax = axes[i]
    ax.imshow(decoded_image.reshape(28, 20), cmap='gray')
    ax.set_title(f'val={val:.1f}', fontsize=8)
    ax.axis('off')

fig.suptitle(f'Traversal of Latent Dimension {latent_dim_to_vary}')
plt.savefig("./data/output/q2/latent_traversal.png")
# plt.show()
plt.close()
