import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

def train_and_generate_gan():
    # Paths
    processed_train_path = "./data/processed/train_cleaned.csv"
    output_dir = "./data/Test"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load data
    print("Loading cleaned dataset...")
    df = pd.read_csv(processed_train_path)
    columns = df.columns.tolist()
    print(f"Data shape: {df.shape}")
    
    # 2. Scale features to [-1, 1] (ideal for Tanh activation in Generator)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(df)
    
    # Convert to PyTorch Tensor
    real_data_tensor = torch.tensor(scaled_data, dtype=torch.float32)
    
    # Hyperparameters
    input_dim = real_data_tensor.shape[1] # 30 features
    noise_dim = 32
    batch_size = 64
    lr = 0.0002
    epochs = 200
    
    # 3. Define Networks
    class Generator(nn.Module):
        def __init__(self, noise_dim, output_dim):
            super(Generator, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(noise_dim, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Linear(256, output_dim),
                nn.Tanh() # Tanh outputs between -1 and 1
            )
        def forward(self, x):
            return self.net(x)

    class Discriminator(nn.Module):
        def __init__(self, input_dim):
            super(Discriminator, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        def forward(self, x):
            return self.net(x)
            
    # Instantiate
    generator = Generator(noise_dim, input_dim)
    discriminator = Discriminator(input_dim)
    
    # Optimizers & Loss
    opt_gen = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_disc = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    
    # DataLoader
    dataset = torch.utils.data.TensorDataset(real_data_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # 4. Training Loop
    print("Training the PyTorch Tabular GAN...")
    hist_d, hist_g = [], []   # per-epoch average losses, for the diagnostic curve
    for epoch in range(epochs):
        ep_d, ep_g, n_batches = 0.0, 0.0, 0
        for batch in loader:
            real_batch = batch[0]
            current_batch_size = real_batch.size(0)

            # --- Train Discriminator ---
            opt_disc.zero_grad()

            # Real samples
            labels_real = torch.ones(current_batch_size, 1) * 0.9 # Label smoothing
            pred_real = discriminator(real_batch)
            loss_disc_real = criterion(pred_real, labels_real)

            # Fake samples
            noise = torch.randn(current_batch_size, noise_dim)
            fake_batch = generator(noise)
            labels_fake = torch.zeros(current_batch_size, 1)
            pred_fake = discriminator(fake_batch.detach())
            loss_disc_fake = criterion(pred_fake, labels_fake)

            loss_disc = loss_disc_real + loss_disc_fake
            loss_disc.backward()
            opt_disc.step()

            # --- Train Generator ---
            opt_gen.zero_grad()
            pred_fake_for_gen = discriminator(fake_batch)
            loss_gen = criterion(pred_fake_for_gen, torch.ones(current_batch_size, 1))
            loss_gen.backward()
            opt_gen.step()

            ep_d += loss_disc.item(); ep_g += loss_gen.item(); n_batches += 1

        hist_d.append(ep_d / n_batches); hist_g.append(ep_g / n_batches)
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}] | Loss D: {hist_d[-1]:.4f} | Loss G: {hist_g[-1]:.4f}")

    print("GAN training completed successfully!")

    # 4b. Save the adversarial loss curve (generator vs. discriminator)
    fig_dir = "./figures"
    os.makedirs(fig_dir, exist_ok=True)
    pd.DataFrame({"epoch": range(1, epochs + 1), "loss_D": hist_d, "loss_G": hist_g}) \
        .to_csv(f"{fig_dir}/gan_loss_history.csv", index=False)
    RED, DARK = "#9B1B30", "#1f1f1f"
    plt.rcParams.update({"font.size": 14, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white",
                         "axes.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.plot(range(1, epochs + 1), hist_d, color=DARK, lw=2, label="Discriminator $D$")
    ax.plot(range(1, epochs + 1), hist_g, color=RED, lw=2, label="Generator $G$")
    ax.set_xlabel("epoch"); ax.set_ylabel("BCE loss")
    ax.set_title("Tabular GAN — adversarial training loss",
                 color=DARK, fontweight="bold")
    ax.grid(alpha=0.25); ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/fig_loss_gan.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig_dir}/fig_loss_gan.png")
    
    # 5. Generate 20,000 samples
    print("Generating 20,000 synthetic samples...")
    generator.eval()
    num_samples = 20000
    with torch.no_grad():
        noise = torch.randn(num_samples, noise_dim)
        generated_scaled = generator(noise).numpy()
        
    # 6. Inverse transform
    generated_raw = scaler.inverse_transform(generated_scaled)
    
    # Convert to DataFrame
    gen_df = pd.DataFrame(generated_raw, columns=columns)
    
    # 7. Post-processing
    # Identify target column
    target_col = 'Class/ASD'
    
    # Identify binary and continuous columns
    # Continuous: age
    # Binary: A1-A10, gender, jaundice, austim, ethnicity_*, relation_*, Class/ASD
    binary_cols = [c for c in columns if c != 'age']
    
    print("Post-processing generated tabular data...")
    for col in binary_cols:
        # Clip and round to 0 or 1
        gen_df[col] = np.clip(np.round(gen_df[col]), 0, 1).astype(int)
        
    # Handle age column (limit between 1 and 80, round to 1 decimal place)
    gen_df['age'] = np.clip(np.round(gen_df['age'], 1), 1.0, 80.0)
    
    # 8. Save output
    output_file = f"{output_dir}/train_cleaned.csv"
    gen_df.to_csv(output_file, index=False)
    print(f"Saved 20,000 synthetic samples to: {output_file}")
    
    # Check class distribution of target
    print("\nGenerated target 'Class/ASD' distribution:")
    print(gen_df[target_col].value_counts(normalize=True) * 100)

if __name__ == "__main__":
    train_and_generate_gan()
