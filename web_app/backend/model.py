"""
Model definitions for the backend.

Includes:
- FixedFeatureExtractor
- HybridColorizationCNN
- BaseColorizationCNN
- color basis utilities and simple preprocess/load helpers

These are minimal definitions enough for inference in the Flask backend.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np

# Model input size constants
IMAGE_DIM = 128
MODEL_WIDTH = IMAGE_DIM
MODEL_HEIGHT = IMAGE_DIM

# Luminance weights for converting RGB to grayscale
LUMINANCE_WEIGHTS = torch.tensor([0.299, 0.587, 0.114])


def compute_orthonormal_basis(w, device='cpu'):
    w = w.to(device)
    w_norm = w / torch.norm(w)
    tmp = torch.tensor([1., 0., 0.], device=device) if abs(w_norm[0]) < 0.9 else torch.tensor([0., 1., 0.], device=device)
    u1 = tmp - w_norm * torch.dot(tmp, w_norm)
    u1 = u1 / (torch.norm(u1) + 1e-8)
    u2 = torch.linalg.cross(w_norm, u1)
    u2 = u2 / (torch.norm(u2) + 1e-8)
    return u1, u2


def reconstruct_color(Y_target, alpha, beta, w, u1, u2):
    w2 = torch.dot(w, w)
    v_parallel = (Y_target / w2).unsqueeze(1) * w
    v_perp = alpha.unsqueeze(1) * u1 + beta.unsqueeze(1) * u2
    return v_parallel + v_perp


class FixedFeatureExtractor(nn.Module):
    """Computes [Sobel_x, Sobel_y, Laplacian, Identity] features from Y.
    Output: [B, 4, H, W]. Kernels are frozen (not trainable).
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=3, padding=1, bias=False)
        self._init_kernels()
        for p in self.parameters():
            p.requires_grad = False

    def _init_kernels(self):
        sobel_x = torch.tensor([
            [-1., 0., 1.],
            [-2., 0., 2.],
            [-1., 0., 1.],
        ])
        sobel_y = torch.tensor([
            [-1., -2., -1.],
            [ 0.,  0.,  0.],
            [ 1.,  2.,  1.],
        ])
        laplacian = torch.tensor([
            [ 0.,  1.,  0.],
            [ 1., -4.,  1.],
            [ 0.,  1.,  0.],
        ])
        identity = torch.tensor([
            [0., 0., 0.],
            [0., 1., 0.],
            [0., 0., 0.],
        ])

        kernels = torch.stack([sobel_x, sobel_y, laplacian, identity], dim=0)  # [4,3,3]
        with torch.no_grad():
            self.conv.weight.copy_(kernels.unsqueeze(1))  # [4,1,3,3]

    def forward(self, Y):
        return self.conv(Y)


class HybridColorizationCNN(nn.Module):
    def __init__(self, m_features=4):
        super().__init__()

        self.m = m_features
        self.fixed_features = FixedFeatureExtractor()
        self.conv1 = nn.Conv2d(1, 32, 4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.tconv1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.tbn1 = nn.BatchNorm2d(64)
        self.tconv2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.tbn2 = nn.BatchNorm2d(32)
        self.tconv3 = nn.ConvTranspose2d(32, 2 * self.m, 4, stride=2, padding=1)

    def forward(self, L):
        with torch.no_grad():
            z = self.fixed_features(L)
        x = F.relu(self.bn1(self.conv1(L)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.tbn1(self.tconv1(x)))
        x = F.relu(self.tbn2(self.tconv2(x)))
        x = self.tconv3(x)
        A, B = torch.chunk(x, 2, dim=1)
        alpha_hat = (A * z).sum(dim=1, keepdim=True)
        beta_hat  = (B * z).sum(dim=1, keepdim=True)
        return alpha_hat, beta_hat


class BaseColorizationCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 4, stride=2, padding=1)  # 32 x H/2 x W/2
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)  # 64 x H/4 x W/4
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 4, stride=2, padding=1) # 128 x H/8 x W/8
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)

        self.tconv1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.tbn1 = nn.BatchNorm2d(64)

        self.tconv2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.tbn2 = nn.BatchNorm2d(32)

        self.tconv3 = nn.ConvTranspose2d(32, 2, 4, stride=2, padding=1)

    def forward(self, L):
        x = F.relu(self.bn1(self.conv1(L)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.tbn1(self.tconv1(x)))
        x = F.relu(self.tbn2(self.tconv2(x)))
        x = self.tconv3(x)   # final alpha,beta
        a = x[:, 0:1]
        b = x[:, 1:2]
        return a, b


def preprocess_image(image_pil):
    # Ensure RGB
    if image_pil.mode != 'RGB':
        image_pil = image_pil.convert('RGB')

    # Resize to model input size
    image_resized = image_pil.resize((MODEL_WIDTH, MODEL_HEIGHT), Image.Resampling.LANCZOS)

    # Convert to numpy array (H, W, 3) with values [0, 255]
    image_array = np.array(image_resized, dtype=np.float32) / 255.0  # (H, W, 3) in [0, 1]

    # Compute luminance: Y = 0.299*R + 0.587*G + 0.114*B
    w = np.array([0.299, 0.587, 0.114], dtype=np.float32)  # (3,)
    luminance = np.dot(image_array, w)  # (H, W)

    # Convert to tensor and add batch/channel dimensions
    tensor = torch.from_numpy(luminance).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    return tensor


def load_model(model_class, checkpoint_path: str, device):
    model = model_class().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"Model loaded from {checkpoint_path}")
    return model
