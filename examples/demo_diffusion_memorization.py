import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import math
import os

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_TRAIN = 500      # Small dataset to force memorization
BATCH_SIZE = 64
LR = 2e-4
EPOCHS = 1500      # Long training to see the phase transition

# --- MODEL: Tiny U-Net ---
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
    def forward(self, x, t):
        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(..., ) + (None, ) * 2]
        h = h + time_emb
        h = self.bnorm2(self.relu(self.conv2(h)))
        return h

class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        image_channels = 3
        down_channels = (32, 64, 128)
        up_channels = (128, 64, 32)
        out_dim = 3 
        time_emb_dim = 32

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        self.downs = nn.ModuleList([])
        for i in range(len(down_channels)-1):
            self.downs.append(nn.ModuleList([
                Block(down_channels[i], down_channels[i], time_emb_dim),
                nn.Conv2d(down_channels[i], down_channels[i+1], 4, 2, 1)
            ]))
        
        self.bottleneck = Block(down_channels[-1], down_channels[-1], time_emb_dim)
        
        self.ups = nn.ModuleList([])
        for i in range(len(up_channels)-1):
            self.ups.append(nn.ModuleList([
                nn.ConvTranspose2d(up_channels[i], up_channels[i+1], 4, 2, 1),
                Block(up_channels[i+1] * 2, up_channels[i+1], time_emb_dim)
            ]))
        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
        x = self.conv0(x)
        residuals = []
        for block, downsample in self.downs:
            x = block(x, t)
            residuals.append(x)
            x = downsample(x)
        x = self.bottleneck(x, t)
        for upsample, block in self.ups:
            x = upsample(x)
            residual = residuals.pop()
            x = torch.cat((x, residual), dim=1) 
            x = block(x, t)
        return self.output(x)

# --- MONITOR ---
class DiffDriftMonitor:
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.activations = {}
        self.layers = {
            'Enc (UV)': model.downs[0][0],
            'Enc (Mid)': model.downs[1][0],
            'Bottle (IR)': model.bottleneck,
            'Dec (Mid)': model.ups[0][1],
            'Dec (UV)': model.ups[1][1]
        }
        for n, m in self.layers.items():
            self.hooks.append(m.register_forward_hook(self.get_hook(n)))
            
    def get_hook(self, name):
        def hook(model, input, output):
            act = output.mean(dim=[2, 3])
            self.activations[name] = act.detach()
        return hook

    def measure_gap(self, x_train, t_train, x_test, t_test):
        _ = self.model(x_train, t_train)
        acts_train = {k: v.clone() for k, v in self.activations.items()}
        _ = self.model(x_test, t_test)
        acts_test = {k: v.clone() for k, v in self.activations.items()}
        
        gaps = {}
        for k in self.layers:
            mu_train = acts_train[k].mean(dim=0)
            mu_test = acts_test[k].mean(dim=0)
            norm = torch.norm(mu_train) + 1e-9
            dist = torch.norm(mu_train - mu_test) / norm
            gaps[k] = dist.item()
        return gaps
    
    def close(self):
        for h in self.hooks: h.remove()

# --- EXECUTION ---
if __name__ == "__main__":
    print(f"🔥 DeepDrift: Diffusion Memorization Experiment (N={N_TRAIN})")
    
    # Data
    os.makedirs('./data', exist_ok=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    train_loader = DataLoader(Subset(dataset, list(range(0, N_TRAIN))), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(Subset(dataset, list(range(N_TRAIN, N_TRAIN + BATCH_SIZE))), batch_size=BATCH_SIZE, shuffle=False)
    
    x_test_fixed, _ = next(iter(test_loader))
    x_test_fixed = x_test_fixed.to(DEVICE)
    x_train_fixed, _ = next(iter(train_loader))
    x_train_fixed = x_train_fixed.to(DEVICE)

    # Model
    model = SimpleUNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    monitor = DiffDriftMonitor(model)
    
    # Diffusion
    T = 300
    betas = torch.linspace(0.0001, 0.02, T).to(DEVICE)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)

    def get_loss(model, x_0, t):
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod[t])[:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1. - alphas_cumprod[t])[:, None, None, None]
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        noise_pred = model(x_t, t)
        return F.mse_loss(noise_pred, noise)

    history_gap = {k: [] for k in monitor.layers}
    history_loss = {'train': [], 'test': []}
    
    # Train Loop
    for epoch in tqdm(range(EPOCHS), desc="Training"):
        model.train()
        for x_batch, _ in train_loader:
            x_batch = x_batch.to(DEVICE)
            t = torch.randint(0, T, (x_batch.shape[0],), device=DEVICE).long()
            optimizer.zero_grad()
            loss = get_loss(model, x_batch, t)
            loss.backward()
            optimizer.step()
            
        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                t_fixed = torch.full((BATCH_SIZE,), T//4, device=DEVICE).long()
                
                # Losses
                l_train = get_loss(model, x_train_fixed, t_fixed).item()
                l_test = get_loss(model, x_test_fixed, t_fixed).item()
                history_loss['train'].append(l_train)
                history_loss['test'].append(l_test)
                
                # Drift Gap
                noise = torch.randn_like(x_train_fixed)
                x_tr_t = torch.sqrt(alphas_cumprod[t_fixed])[:,None,None,None] * x_train_fixed + \
                         torch.sqrt(1 - alphas_cumprod[t_fixed])[:,None,None,None] * noise
                x_te_t = torch.sqrt(alphas_cumprod[t_fixed])[:,None,None,None] * x_test_fixed + \
                         torch.sqrt(1 - alphas_cumprod[t_fixed])[:,None,None,None] * noise
                
                gaps = monitor.measure_gap(x_tr_t, t_fixed, x_te_t, t_fixed)
                for k, v in gaps.items():
                    history_gap[k].append(v)
    
    monitor.close()
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    ax = axes[0]
    ax.plot(history_loss['train'], label='Train Loss', color='blue')
    ax.plot(history_loss['test'], label='Test Loss', color='orange', linestyle='--')
    ax.set_title('Generalization vs Memorization Gap')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    data = np.array([history_gap[k] for k in monitor.layers])
    sns.heatmap(data, ax=ax, cmap="magma", yticklabels=list(monitor.layers.keys()))
    ax.set_title('Spatial Anatomy of Memorization (DeepDrift)')
    ax.set_xlabel('Time (x20 Epochs)')
    
    plt.tight_layout()
    plt.savefig('diffusion_memorization.png')
    print("✅ Experiment Complete. Results saved to 'diffusion_memorization.png'")
