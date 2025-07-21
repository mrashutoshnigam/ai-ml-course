import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from scipy.io import loadmat
import os
from skimage.metrics import peak_signal_noise_ratio as psnr

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# U-Net Autoencoder Architecture (without skip connections)


class UNetAutoencoder(nn.Module):
    def __init__(self):
        super(UNetAutoencoder, self).__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU())
        self.enc4 = nn.Linear(256 * 7 * 7, 128)  # Latent space
        # Decoder
        self.dec1 = nn.Linear(128, 256 * 7 * 7)
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(
            256, 128, 3, stride=2, padding=1, output_padding=1), nn.ReLU())
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(
            128, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU())
        self.dec4 = nn.Sequential(nn.Conv2d(64, 1, 3, padding=1), nn.Sigmoid())

    def encode(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = x.view(x.size(0), -1)
        return self.enc4(x)

    def decode(self, h):
        x = self.dec1(h)
        x = x.view(x.size(0), 256, 7, 7)
        x = self.dec2(x)
        x = self.dec3(x)
        return self.dec4(x)

    def forward(self, x):
        h = self.encode(x)
        return self.decode(h), h

# Sparse Autoencoder


class SparseAutoencoder(UNetAutoencoder):
    def __init__(self, sparsity_param=0.1, sparsity_weight=3e-3):
        super(SparseAutoencoder, self).__init__()
        self.sparsity_param = sparsity_param
        self.sparsity_weight = sparsity_weight

    def sparsity_loss(self, h):
        rho = torch.mean(h, dim=0)
        rho_hat = self.sparsity_param
        kl_div = rho_hat * \
            torch.log(rho_hat / rho) + (1 - rho_hat) * \
            torch.log((1 - rho_hat) / (1 - rho))
        return self.sparsity_weight * torch.sum(kl_div)

# Contractive Autoencoder


class ContractiveAutoencoder(UNetAutoencoder):
    def __init__(self, contractive_weight=1e-4):
        super(ContractiveAutoencoder, self).__init__()
        self.contractive_weight = contractive_weight

    def contractive_loss(self, x, h):
        h.backward(torch.ones_like(h), retain_graph=True)
        contractive_loss = torch.mean(torch.sum(x.grad ** 2, dim=(1, 2, 3)))
        x.grad = None
        return self.contractive_weight * contractive_loss

# Variational Autoencoder


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dim=20):
        super(VariationalAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU())
        self.fc_mu = nn.Linear(256 * 5 * 7, latent_dim)
        self.fc_logvar = nn.Linear(256 * 5 * 7, latent_dim)
        # Decoder
        self.dec1 = nn.Linear(latent_dim, 256 * 5 * 7)
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(
            256, 128, 3, stride=2, padding=1, output_padding=1), nn.ReLU())
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(
            128, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU())
        self.dec4 = nn.Sequential(nn.Conv2d(64, 1, 3, padding=1), nn.Sigmoid())

    def encode(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.dec1(z)
        x = x.view(x.size(0), 256, 5, 7)
        x = self.dec2(x)
        x = self.dec3(x)
        return self.dec4(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# Data Loading
transform = transforms.Compose([transforms.ToTensor()])
mnist_train = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(mnist_train, batch_size=128, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=128, shuffle=False)

# Frey Face dataset (assumed to be in data/train and data/test)


def load_frey_face():
    data = loadmat('data/frey_rawface.mat')['ff'].T / 255.0
    data = data.reshape(-1, 1, 28, 20).astype(np.float32)
    return torch.utils.data.TensorDataset(torch.tensor(data))


frey_dataset = load_frey_face()
frey_loader = DataLoader(frey_dataset, batch_size=128, shuffle=True)

# Training Functions


def train_sparse_ae(model, loader, epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, _ in loader:
            x = x.to(device)
            optimizer.zero_grad()
            recon, h = model(x)
            recon_loss = criterion(recon, x)
            sparsity_loss = model.sparsity_loss(h)
            loss = recon_loss + sparsity_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Sparse AE Epoch {epoch+1}, Loss: {total_loss/len(loader)}')


def train_contractive_ae(model, loader, epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, _ in loader:
            x = x.to(device)
            x.requires_grad_(True)
            optimizer.zero_grad()
            recon, h = model(x)
            recon_loss = criterion(recon, x)
            contractive_loss = model.contractive_loss(x, h)
            loss = recon_loss + contractive_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(
            f'Contractive AE Epoch {epoch+1}, Loss: {total_loss/len(loader)}')


def train_vae(model, loader, epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, in loader:
            x = x.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            recon_loss = nn.functional.binary_cross_entropy(
                recon, x, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'VAE Epoch {epoch+1}, Loss: {total_loss/len(loader)}')

# Task 1(a): t-SNE Visualization


def plot_tsne(model, loader, title, filename):
    model.eval()
    embeddings, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            h = model.encode(x)
            embeddings.append(h.cpu().numpy())
            labels.append(y.numpy())
    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(embeddings)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='tab10', s=5)
    plt.colorbar(scatter)
    plt.title(title)
    plt.savefig(filename)
    plt.close()

# Task 1(b): Interpolation and PSNR


def interpolation_analysis(model, loader, filename):
    model.eval()
    data_iter = iter(loader)
    results = []
    for _ in range(20):
        (x1, y1), (x2, y2) = next(data_iter), next(data_iter)
        while y1[0] == y2[0]:
            (x2, y2) = next(data_iter)
        x1, x2 = x1[0:1].to(device), x2[0:1].to(device)
        h1 = model.encode(x1)
        h2 = model.encode(x2)
        alphas = [0, 0.2, 0.4, 0.6, 0.8, 1]
        fig, axes = plt.subplots(2, len(alphas), figsize=(15, 5))
        for i, alpha in enumerate(alphas):
            I_alpha = alpha * x1 + (1 - alpha) * x2
            h_alpha = model.encode(I_alpha)
            h_prime_alpha = alpha * h1 + (1 - alpha) * h2
            I_hat_alpha = model.decode(h_alpha)
            I_hat_prime_alpha = model.decode(h_prime_alpha)
            psnr_val = psnr(I_hat_alpha.cpu().detach().numpy(
            ), I_hat_prime_alpha.cpu().detach().numpy(), data_range=1)
            l2_norm = torch.norm(h_alpha - h_prime_alpha).item()
            results.append((alpha, psnr_val, l2_norm))
            axes[0, i].imshow(
                I_hat_alpha.cpu().squeeze().detach().numpy(), cmap='gray')
            axes[0, i].set_title(f'α={alpha}')
            axes[1, i].imshow(I_hat_prime_alpha.cpu(
            ).squeeze().detach().numpy(), cmap='gray')
            axes[1, i].axis('off')
        plt.savefig(f'{filename}_{_}.png')
        plt.close()
    return results

# Task 1(c): Classification


def classify_embeddings(model, train_loader, test_loader):
    model.eval()
    X_train, y_train = [], []
    X_test, y_test = [], []
    with torch.no_grad():
        for x, y in train_loader:
            x = x.to(device)
            h = model.encode(x)
            X_train.append(h.cpu().numpy())
            y_train.append(y.numpy())
        for x, y in test_loader:
            x = x.to(device)
            h = model.encode(x)
            X_test.append(h.cpu().numpy())
            y_test.append(y.numpy())
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    X_test = np.concatenate(X_test)
    y_test = np.concatenate(y_test)
    clf = SVC(kernel='rbf')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred)

# Task 2: VAE Sampling


def sample_vae(model, filename):
    model.eval()
    with torch.no_grad():
        z = torch.randn(10, 20).to(device)
        for i in range(20):
            z_perturbed = z.clone()
            z_perturbed[:, i] += torch.linspace(-2, 2, 10).to(device)
            samples = model.decode(z_perturbed)
            fig, axes = plt.subplots(1, 10, figsize=(20, 2))
            for j in range(10):
                axes[j].imshow(samples[j].cpu().squeeze().numpy(), cmap='gray')
                axes[j].axis('off')
            plt.savefig(f'{filename}_latent_{i}.png')
            plt.close()


# Main Execution
sparse_ae = SparseAutoencoder().to(device)
contractive_ae = ContractiveAutoencoder().to(device)
vae = VariationalAutoencoder().to(device)

# Train models
train_sparse_ae(sparse_ae, train_loader)
train_contractive_ae(contractive_ae, train_loader)
train_vae(vae, frey_loader)

# Task 1(a)
plot_tsne(sparse_ae, test_loader, 'Sparse AE t-SNE', 'sparse_tsne.png')
plot_tsne(contractive_ae, test_loader,
          'Contractive AE t-SNE', 'contractive_tsne.png')

# Task 1(b)
sparse_results = interpolation_analysis(
    sparse_ae, test_loader, 'sparse_interpolation')
contractive_results = interpolation_analysis(
    contractive_ae, test_loader, 'contractive_interpolation')

# Task 1(c)
sparse_accuracy = classify_embeddings(sparse_ae, train_loader, test_loader)
contractive_accuracy = classify_embeddings(
    contractive_ae, train_loader, test_loader)

# Task 2
sample_vae(vae, 'vae_samples')

# Save results for report
with open('results.txt', 'w') as f:
    f.write(f'Sparse AE Accuracy: {sparse_accuracy}\n')
    f.write(f'Contractive AE Accuracy: {contractive_accuracy}\n')
    f.write('Sparse AE Interpolation Results:\n')
    for alpha, psnr_val, l2_norm in sparse_results:
        f.write(f'Alpha: {alpha}, PSNR: {psnr_val}, L2 Norm: {l2_norm}\n')
    f.write('Contractive AE Interpolation Results:\n')
    for alpha, psnr_val, l2_norm in contractive_results:
        f.write(f'Alpha: {alpha}, PSNR: {psnr_val}, L2 Norm: {l2_norm}\n')
