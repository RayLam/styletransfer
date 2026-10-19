import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.models as models

# -----------------------------------------------------------------------------
# 1. Image Path Definition
# -----------------------------------------------------------------------------
IMAGE_PATH = "/Users/raylam/styletransfer/846910042_122199315722399604_3034258827721968700_n.jpg"
OUTPUT_DUMP_PATH = "style_features_gram_dump.txt"

# -----------------------------------------------------------------------------
# 2. VGG-19 Layer Feature Extractor Architecture
# -----------------------------------------------------------------------------
class StyleFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        
        # Slicing pre-trained VGG19 layers for multi-scale style capture
        self.slice1 = vgg[:2]    # conv1_1 (Low-level colors and fine lines)
        self.slice2 = vgg[2:7]   # conv2_1 (Medium brushstrokes & textures)
        self.slice3 = vgg[7:12]  # conv3_1 (Complex curves & color gradients)
        self.slice4 = vgg[12:21] # conv4_1 (Structural style templates)
        
        # Freeze network parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        return {"conv1_1": h1, "conv2_1": h2, "conv3_1": h3, "conv4_1": h4}


def compute_gram_matrix(feature_map: torch.Tensor) -> torch.Tensor:
    """Computes normalized Gram Matrix G = (F * F^T) / (C * H * W)"""
    b, c, h, w = feature_map.size()
    
    # Flatten spatial grid [B, C, H, W] -> [C, H * W]
    features = feature_map.view(c, h * w)
    
    # Compute dot product correlation matrix [C, C]
    gram = torch.mm(features, features.t())
    
    # Normalize by total spatial volume to avoid scale explosion
    return gram / (c * h * w)


# -----------------------------------------------------------------------------
# 3. Execution Pipeline & File Dump Writer
# -----------------------------------------------------------------------------
def run_step_4_style_dump(image_path: str, output_txt: str):
    print(f"Loading reference mural from: {image_path}")
    
    # Load and normalize raw RGB pixels [0.0, 1.0]
    pil_img = Image.open(image_path).convert("RGB")
    hwc_array = np.array(pil_img, dtype=np.float32) / 255.0
    raw_H, raw_W, C = hwc_array.shape
    
    # Convert [H, W, C] -> PyTorch Batch Tensor [1, C, H, W]
    tensor_batch = torch.from_numpy(hwc_array).permute(2, 0, 1).unsqueeze(0)
    
    # Auto-pad bottom and right edges so H and W are cleanly divisible by 16
    pad_h = (16 - (raw_H % 16)) % 16
    pad_w = (16 - (raw_W % 16)) % 16
    padded_batch = F.pad(tensor_batch, (0, pad_w, 0, pad_h), mode="reflect")
    _, _, H, W = padded_batch.shape
    
    print(f"Processed Tensor Shape: [1, {C}, {H}, {W}] (Pad: +{pad_h}px H, +{pad_w}px W)\n")

    # ImageNet Standard Normalization (Mean & Std subtraction for VGG19)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    normalized_input = (padded_batch - mean) / std

    # Pass through Deep Feature Extractor
    extractor = StyleFeatureExtractor()
    layer_features = extractor(normalized_input)
    
    # Write feature shapes and Gram Matrix numerical values to text file
    with open(output_txt, "w") as f:
        f.write(f"=== LAYER-WISE STYLE STATISTICS DUMP FOR FILE: {image_path} ===\n")
        f.write(f"Image Resolution: {H}x{W} | Padded Divisible by 16: True\n\n")

        for layer_name, fmap in layer_features.items():
            gram = compute_gram_matrix(fmap).numpy()
            c_dim = gram.shape[0]
            
            f.write(f"==================================================\n")
            f.write(f"LAYER: {layer_name} | FEATURE MAP SHAPE: {list(fmap.shape)}\n")
            f.write(f"GRAM MATRIX SHAPE: [{c_dim} x {c_dim}] ({c_dim * c_dim:,} Correlation Entries)\n")
            f.write(f"--------------------------------------------------\n")
            f.write(f"Gram Matrix Range: Min = {gram.min():.6f}, Max = {gram.max():.6f}, Mean = {gram.mean():.6f}\n")
            f.write(f"Top-Left 4x4 Corner Slice:\n{np.array2string(gram[:4, :4], precision=6, floatmode='fixed')}\n\n")
            
            # Dump full Gram matrix values
            f.write(f"FULL GRAM MATRIX VALUES [{c_dim}x{c_dim}]:\n")
            for r_idx in range(c_dim):
                row_vals = ", ".join([f"{val:.6f}" for val in gram[r_idx]])
                f.write(f"Row {r_idx:03d}: [{row_vals}]\n")
            f.write("\n")

    print(f"SUCCESS! Style Gram Matrices exported to '{output_txt}'.")


if __name__ == "__main__":
    run_step_4_style_dump(IMAGE_PATH, OUTPUT_DUMP_PATH)