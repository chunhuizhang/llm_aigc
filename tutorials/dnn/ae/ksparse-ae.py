import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F

# 设置随机种子和设备
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据加载和预处理
transform = transforms.Compose([
    # 0-1 之间
    transforms.ToTensor(),
    # 这种运行不会出错的，数据/逻辑bug，很难排查
    # transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

class KSparseAutoencoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, k=60):
        super(KSparseAutoencoder, self).__init__()
        self.k = k
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            # nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def k_sparse(self, x):
        # 实现k-sparse约束
        topk, indices = torch.topk(x, self.k, dim=1)
        mask = torch.zeros_like(x).scatter_(1, indices, 1)
        return x * mask
    
    def forward(self, x):
        # 编码
        x = x.view(x.size(0), -1)
        encoded = self.encoder(x)
        # 应用k-sparse约束
        sparse_encoded = self.k_sparse(encoded)
        # 解码
        decoded = self.decoder(sparse_encoded)
        return decoded, sparse_encoded

def train_model(model, train_loader, num_epochs=20):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    train_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for data, _ in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            data = data.to(device)
            optimizer.zero_grad()
            
            decoded, _ = model(data)
            loss = criterion(decoded, data.view(data.size(0), -1))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}')
    
    return train_losses

def visualize_results(model, test_loader, num_images=10, k=20):
    model.eval()
    with torch.no_grad():
        data = next(iter(test_loader))[0][:num_images].to(device)
        labels = next(iter(test_loader))[1][:num_images]
        decoded, encoded = model(data)
        
        # 可视化原始图像和重构图像
        fig, axes = plt.subplots(2, num_images, figsize=(20, 4))
        
        for i in range(num_images):
            # 原始图像
            err = F.mse_loss(data[i].cpu().squeeze(), decoded[i].cpu().view(28, 28))
            print(labels[i].item(), err.item())
            axes[0, i].imshow(data[i].cpu().squeeze(), cmap='gray')
            axes[0, i].axis('off')
            axes[0, i].set_title('Original')
            
            # 重构图像
            axes[1, i].imshow(decoded[i].cpu().view(28, 28), cmap='gray')
            axes[1, i].axis('off')
        
            axes[1, i].set_title(f'Reconstructed, {err.item():.2f}')
        
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'./figs/ksae-reconstructed-{k}.png')
        
        # 可视化隐层激活
        plt.figure(figsize=(10, 4))
        plt.imshow(encoded.cpu().T, aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.title('Hidden Layer Activations')
        plt.xlabel('Sample')
        plt.ylabel('Hidden Unit')
        # plt.show()
        plt.savefig(f'./figs/ksae-activations-{k}.png')

def plot_training_loss(losses, k):
    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.title(f'Training Loss Over Time (k={k})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    # plt.show()
    plt.savefig(f'./figs/ksae-loss-{k}.png')

# 主函数
def main():
    k = 20
    model = KSparseAutoencoder(k=k).to(device)
    losses = train_model(model, train_loader)
    plot_training_loss(losses, k)
    visualize_results(model, test_loader, k=k)

if __name__ == "__main__":
    main()
