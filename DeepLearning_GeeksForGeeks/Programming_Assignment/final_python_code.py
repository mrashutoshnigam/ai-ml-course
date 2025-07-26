import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import math
import os

# Tensorflow setup to use GPU if available
tf.random.set_seed(42)
print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(
            f"Found {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found, using CPU instead")
if gpus:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("Mixed precision enabled")

# Hyperparameters
batch_size = 128
epochs = 10
learning_rate = 0.001
sparse_lambda = 1e-3
contractive_lambda = 1e-4
rho = 0.05
epsilon = 1e-6

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)
train_dataset = tf.data.Dataset.from_tensor_slices(x_train).cache().shuffle(
    60000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_dataset = tf.data.Dataset.from_tensor_slices(
    (x_test, y_test)).batch(batch_size).prefetch(tf.data.AUTOTUNE)


# U-Net-like Encoder and decoder for MNIST
def build_encoder():
    inputs = layers.Input(shape=(28, 28, 1))
    x = layers.Conv2D(64, 3, padding='same',
                      activation='relu')(inputs)  # 28x28x64
    x = layers.MaxPooling2D(2)(x)  # 14x14x64
    x = layers.Conv2D(128, 3, padding='same',
                      activation='relu')(x)  # 14x14x128
    x = layers.MaxPooling2D(2)(x)  # 7x7x128
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)  # 7x7x256
    x = layers.MaxPooling2D(2)(x)  # 3x3x256
    x = layers.Flatten()(x)
    z = layers.Dense(128)(x)  # Latent space
    return models.Model(inputs, z, name='encoder')


def build_decoder():
    inputs = layers.Input(shape=(128,))
    x = layers.Dense(256 * 3 * 3, activation='relu')(inputs)
    x = layers.Reshape((3, 3, 256))(x)
    x = layers.Conv2DTranspose(
        128, 3, strides=2, padding='same', activation='relu')(x)  # 6x6x128
    x = layers.Conv2DTranspose(
        64, 3, strides=2, padding='same', activation='relu')(x)  # 12x12x64
    x = layers.Conv2DTranspose(
        1, 3, strides=2, padding='same', activation='sigmoid')(x)  # 28x28x1
    x = layers.Conv2D(1, 3, padding='valid',
                      activation='sigmoid')(x)  # 22x22x1
    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(x)  # 28x28x1
    return models.Model(inputs, x, name='decoder')


# Sparse Autoencoder
class SparseAutoencoder(models.Model):
    def __init__(self):
        super(SparseAutoencoder, self).__init__()
        self.encoder = build_encoder()
        self.decoder = build_decoder()

    def call(self, inputs):
        z = self.encoder(inputs)
        recon = self.decoder(z)
        return recon, z

    def sparse_ae_loss(self, y_true, y_pred, z):
        mse_loss = tf.reduce_mean(tf.keras.losses.mse(y_true, y_pred))
        rho_hat = tf.reduce_mean(z, axis=0)
        rho_hat = tf.clip_by_value(rho_hat, epsilon, 1 - epsilon)
        kl_div = rho * tf.math.log(rho / rho_hat) + \
            (1 - rho) * tf.math.log((1 - rho) / (1 - rho_hat))
        # Clip KL term
        kl_loss = sparse_lambda * tf.reduce_sum(
            tf.clip_by_value(kl_div, -1e4, 1e4))
        return mse_loss + kl_loss


class ContractiveAutoencoder(models.Model):
    def __init__(self):
        super(ContractiveAutoencoder, self).__init__()
        self.encoder = build_encoder()
        self.decoder = build_decoder()

    def call(self, inputs):
        z = self.encoder(inputs)
        recon = self.decoder(z)
        return recon, z

    def contractive_ae_loss(self, x, recon, z, model):
        mse_loss = tf.reduce_mean(tf.square(x - recon))
        with tf.GradientTape() as tape:
            tape.watch(z)
            recon = model.decoder(z)
        grad_z = tape.gradient(recon, z)
        j_loss = contractive_lambda * tf.reduce_mean(
            tf.reduce_sum(tf.square(grad_z), axis=1))
        return mse_loss + j_loss


# Train function for both Sparse and Contractive Autoencoder
def train(model, dataset, loss_fn, epochs, model_type='sparse'):
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataset:
            with tf.GradientTape() as tape:
                recon, z = model(batch)
                if model_type == 'sparse':
                    loss = loss_fn(batch, recon, z)
                else:
                    loss = loss_fn(batch, recon, z, model)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(
                zip(gradients, model.trainable_variables))
            total_loss += loss.numpy()
        print(
            f'Epoch {epoch+1}/{epochs}, {model_type.capitalize()} AE Loss: {total_loss / len(dataset): .6f}')


def reconstruct_and_validate(model, dataset, model_name, num_images=5):
    for batch in dataset.take(1):
        images, label = batch
        original_images = images.numpy()
        recon_images, _ = model(original_images, training=False)
        recon_images = recon_images.numpy()

    # Compute MSE for the batch
    mse = np.mean((original_images - recon_images) ** 2)
    print(f"{model_name} Test MSE: {mse:.6f}")

    # Visualize original and reconstructed images
    plt.figure(figsize=(10, 4))
    for i in range(num_images):
        plt.subplot(2, num_images, i + 1)
        plt.imshow(original_images[i].reshape(28, 28), cmap='gray')
        plt.title(f"Original {i+1}")
        plt.axis('off')
        plt.subplot(2, num_images, i + num_images + 1)
        plt.imshow(recon_images[i].reshape(28, 28), cmap='gray')
        plt.title(f"Reconstructed {i+1}")
        plt.axis('off')
    plt.suptitle(f"{model_name} Reconstruction")
    plt.tight_layout()
    plt.savefig(f"./data/output/{model_name}_reconstruction.png")
    plt.show()

# Function to plot t-SNE of embeddings


def plot_tsne_embeddings(model, dataset, model_name, num_samples=1000):

    embeddings = []
    labels = []
    for batch_images, batch_labels in dataset.take(num_samples // batch_size + 1):
        z = model.encoder(batch_images, training=False).numpy()
        embeddings.append(z)
        labels.append(batch_labels.numpy())
    embeddings = np.concatenate(embeddings, axis=0)[:num_samples]
    labels = np.concatenate(labels, axis=0)[:num_samples]

    # Apply t-SNE to reduce to 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300)
    tsne_embeddings = tsne.fit_transform(embeddings)

    # Plot t-SNE with colors for each class
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=tsne_embeddings[:, 0], y=tsne_embeddings[:, 1], hue=labels,
                    palette='tab10', legend='full', s=50)
    plt.title(f"t-SNE of {model_name} Embeddings (MNIST Classes)")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(title="Digit", loc="best")
    plt.tight_layout()
    plt.savefig(f"./data/output/{model_name}_tsne.png")
    plt.show()


def compute_psnr(img1, img2, max_val=1.0):
    mse = mean_squared_error(img1.flatten(), img2.flatten())
    if mse == 0:
        return float('inf')
    return 20 * math.log10(max_val / math.sqrt(mse))

# Function to select pairs and perform interpolation analysis


def interpolation_analysis(model, dataset, model_name, num_pairs=20, num_images_per_pair=5):
    # Collect images and labels from test set
    all_images, all_labels = [], []
    for images, labels in dataset:
        all_images.append(images.numpy())
        all_labels.append(labels.numpy())
    all_images = np.concatenate(all_images, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Randomly select pairs from different classes
    np.random.seed(42)  # For reproducibility
    indices = np.arange(len(all_labels))
    np.random.shuffle(indices)
    selected_pairs = []
    used_labels = set()
    i = 0
    while len(selected_pairs) < num_pairs and i < len(indices):
        idx1 = indices[i]
        label1 = all_labels[idx1]
        # Find an index with a different label
        for j in range(i + 1, len(indices)):
            idx2 = indices[j]
            label2 = all_labels[idx2]
            if label1 != label2 and (label1, label2) not in used_labels:
                selected_pairs.append((idx1, idx2))
                used_labels.add((label1, label2))
                break
        i += 1

    # Alpha values for interpolation
    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    # Store metrics
    psnr_values = {alpha: [] for alpha in alphas}
    l2_norms = {alpha: [] for alpha in alphas}

    # Process each pair
    for pair_idx, (idx1, idx2) in enumerate(selected_pairs[:num_pairs]):
        I1 = all_images[idx1:idx1+1]  # Shape: (1, 28, 28, 1)
        I2 = all_images[idx2:idx2+1]
        label1, label2 = all_labels[idx1], all_labels[idx2]

        # Get embeddings h1 and h2
        h1 = model.encoder(I1, training=False).numpy()  # Shape: (1, 64)
        h2 = model.encoder(I2, training=False).numpy()

        # Plot for this pair
        plt.figure(figsize=(12, len(alphas) * 2))
        for i, alpha in enumerate(alphas):
            # Compute interpolated image Iα
            I_alpha = alpha * I1 + (1 - alpha) * I2
            # Compute embedding hα = E(Iα)
            h_alpha = model.encoder(I_alpha, training=False).numpy()
            # Compute approximate embedding h′α = αh1 + (1−α)h2
            h_prime_alpha = alpha * h1 + (1 - alpha) * h2
            # Decode to get Îα and Î′α
            I_hat_alpha = model.decoder(h_alpha, training=False).numpy()
            I_hat_prime_alpha = model.decoder(
                h_prime_alpha, training=False).numpy()

            # Compute metrics
            psnr = compute_psnr(I_hat_alpha[0], I_hat_prime_alpha[0])
            l2_norm = np.sqrt(np.sum((h_alpha - h_prime_alpha) ** 2))
            psnr_values[alpha].append(psnr)
            l2_norms[alpha].append(l2_norm)

            # Plot Îα and Î′α
            plt.subplot(len(alphas), 2, i * 2 + 1)
            plt.imshow(I_hat_alpha[0].reshape(28, 28), cmap='gray')
            plt.title(f"Îα (α={alpha:.1f})")
            plt.axis('off')
            plt.subplot(len(alphas), 2, i * 2 + 2)
            plt.imshow(I_hat_prime_alpha[0].reshape(28, 28), cmap='gray')
            plt.title(f"Î′α (α={alpha:.1f})")
            plt.axis('off')

        plt.suptitle(
            f"{model_name} Pair {pair_idx+1}: Digit {label1} to {label2}")
        plt.tight_layout()
        plt.savefig(f"./data/output/{model_name}_pair_{pair_idx+1}.png")
        plt.show()

    # Report average metrics
    print(f"\n{model_name} Metrics:")
    print("Alpha | Avg PSNR (dB) | Avg L2 Norm")
    print("------|---------------|------------")
    for alpha in alphas:
        avg_psnr = np.mean(psnr_values[alpha])
        avg_l2_norm = np.mean(l2_norms[alpha])
        print(f"{alpha:.1f}  | {avg_psnr:.4f}      | {avg_l2_norm:.4f}")


# Function to perform classification and evaluate embeddings
def classify_embeddings(model, train_data, train_labels, test_data, test_labels, model_name):
    # Extract embeddings for training set
    train_embeddings = []
    for batch in tf.data.Dataset.from_tensor_slices(train_data).batch(batch_size):
        z = model.encoder(batch, training=False).numpy()
        train_embeddings.append(z)
    train_embeddings = np.concatenate(train_embeddings, axis=0)

    # Extract embeddings for test set
    test_embeddings = []
    for batch, _ in test_dataset:
        z = model.encoder(batch, training=False).numpy()
        test_embeddings.append(z)
    test_embeddings = np.concatenate(test_embeddings, axis=0)

    # Train logistic regression classifier
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(train_embeddings, train_labels)

    # Predict on test embeddings
    test_predictions = classifier.predict(test_embeddings)

    # Compute accuracy
    accuracy = accuracy_score(test_labels, test_predictions)
    print(f"{model_name} Classification Accuracy: {accuracy:.4f}")

    return accuracy


# Question 1(a): Build Sparse and Contractive Autoencoders
sparse_ae = SparseAutoencoder()
contractive_ae = ContractiveAutoencoder()
print("Sparse Autoencoder Summary:")
sparse_ae.summary()
print("\nContractive Autoencoder Summary:")
contractive_ae.summary()

# Train Sparse Autoencoder
print("Training Sparse Autoencoder...")
train(sparse_ae, train_dataset, sparse_ae.sparse_ae_loss,
      epochs, model_type='sparse')

# Train Contractive Autoencoder
print("\nTraining Contractive Autoencoder...")
train(contractive_ae, train_dataset, contractive_ae.contractive_ae_loss,
      epochs, model_type='contractive')

print("\nReconstructing and Validating Sparse Autoencoder...")
reconstruct_and_validate(sparse_ae, test_dataset, "Sparse Autoencoder")

print("\nReconstructing and Validating Contractive Autoencoder...")
reconstruct_and_validate(contractive_ae, test_dataset,
                         "Contractive Autoencoder")

# After training and validation, plot t-SNE
print("\nPlotting t-SNE for Sparse Autoencoder...")
plot_tsne_embeddings(sparse_ae, test_dataset, "Sparse Autoencoder")

print("\nPlotting t-SNE for Contractive Autoencoder...")
plot_tsne_embeddings(contractive_ae, test_dataset, "Contractive Autoencoder")

# After training and other validations, perform interpolation analysis
print("\nInterpolation Analysis for Sparse Autoencoder...")
interpolation_analysis(sparse_ae, test_dataset, "Sparse Autoencoder")

print("\nInterpolation Analysis for Contractive Autoencoder...")
interpolation_analysis(contractive_ae, test_dataset, "Contractive Autoencoder")


# After training, reconstruction, t-SNE, and interpolation analysis
print("\nClassifying Digits using Sparse Autoencoder Embeddings...")
sparse_accuracy = classify_embeddings(
    sparse_ae, x_train, y_train, x_test, y_test, "Sparse Autoencoder")

print("\nClassifying Digits using Contractive Autoencoder Embeddings...")
contractive_accuracy = classify_embeddings(
    contractive_ae, x_train, y_train, x_test, y_test, "Contractive Autoencoder")

# Compare and report which is better
print("\nComparison:")
if sparse_accuracy > contractive_accuracy:
    print(
        f"Sparse Autoencoder is better with accuracy {sparse_accuracy:.4f} vs. Contractive Autoencoder {contractive_accuracy:.4f}")
elif contractive_accuracy > sparse_accuracy:
    print(
        f"Contractive Autoencoder is better with accuracy {contractive_accuracy:.4f} vs. Sparse Autoencoder {sparse_accuracy:.4f}")
else:
    print(f"Both autoencoders have equal accuracy: {sparse_accuracy:.4f}")
