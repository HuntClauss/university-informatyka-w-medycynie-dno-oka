import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from skimage.filters import frangi
from sklearn.metrics import jaccard_score, f1_score

# ----------------- COPIED ARCHITECTURE -----------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.attns = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        curr_in = in_channels
        for feature in features:
            self.downs.append(DoubleConv(curr_in, feature))
            curr_in = feature
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.attns.append(AttentionGate(F_g=feature, F_l=feature, F_int=max(16, feature//2)))
            self.ups.append(DoubleConv(feature*2, feature))
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            attn_skip = self.attns[idx//2](g=x, x=skip_connection)
            concat_x = torch.cat((attn_skip, x), dim=1)
            x = self.ups[idx+1](concat_x)
        return self.final_conv(x)

# ----------------- MAIN RUNNER -----------------
def main():
    DATA_DIR = Path("data")
    ORIGINAL_DIR = DATA_DIR / "original"
    LABELS_DIR = DATA_DIR / "labels"
    
    all_image_ids = sorted([p.stem for p in ORIGINAL_DIR.glob("*.ppm")])
    val_ids = all_image_ids[15:]
    
    # Instantiate & Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionUNet(in_channels=3, out_channels=1, features=[64, 128, 256, 512]).to(device)
    model.load_state_dict(torch.load("best_unet_small_vessel.pth", map_location=device))
    model.eval()
    
    # Target Artifact Directory
    artifact_dir = Path("/home/redsec/.gemini/antigravity-ide/brain/5b9f058e-87a8-4f6d-8067-be0eae391f92")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    # Best threshold from evaluation was 0.40
    best_threshold = 0.40
    
    print("Generating validation visualizations...")
    with torch.no_grad():
        for i, img_id in enumerate(val_ids):
            img_path = ORIGINAL_DIR / f"{img_id}.ppm"
            lbl_path = LABELS_DIR / f"{img_id}.vk.ppm"
            
            # Preprocess
            image_np = np.array(Image.open(img_path).convert("RGB"))
            label_np = np.array(Image.open(lbl_path).convert("L"))
            
            green = image_np[:, :, 1]
            enhanced_green = clahe.apply(green)
            
            enhanced_green_norm = enhanced_green.astype(float) / 255.0
            frangi_map = frangi(enhanced_green_norm, sigmas=(0.5, 1.0, 1.5, 2.0), black_ridges=True)
            if frangi_map.max() > 0:
                frangi_map = frangi_map / frangi_map.max()
                
            stacked = np.stack([
                enhanced_green.astype(np.float32) / 255.0,
                frangi_map.astype(np.float32),
                green.astype(np.float32) / 255.0
            ], axis=-1)
            
            img_tensor = torch.from_numpy(stacked).permute(2, 0, 1)
            
            # Pad to 704x608
            val_padding = (2, 1, 2, 2) # left, top, right, bottom
            img_pad = TF.pad(img_tensor, val_padding, fill=0)
            
            # Predict
            img_tensor_batch = img_pad.unsqueeze(0).to(device)
            outputs = model(img_tensor_batch)
            prob_map_full = torch.sigmoid(outputs).squeeze(0).squeeze(0).cpu().numpy()
            pred_mask_full = (prob_map_full > best_threshold).astype(np.uint8)
            
            # Crop back to 700x605
            prob_map = prob_map_full[1:606, 2:702]
            pred_mask = pred_mask_full[1:606, 2:702]
            
            # Metrics
            flat_mask = (label_np.ravel() > 127).astype(np.uint8)
            flat_pred = pred_mask.ravel()
            iou = jaccard_score(flat_mask, flat_pred, zero_division=0)
            f1 = f1_score(flat_mask, flat_pred, zero_division=0)
            
            # Plot
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            fig.suptitle(f"Validation Sample: {img_id} | IoU = {iou:.4f} | F1 = {f1:.4f} (Thresh: {best_threshold:.2f})", fontsize=14)
            
            axes[0].imshow(enhanced_green, cmap="gray")
            axes[0].set_title("CLAHE Green Input")
            axes[0].axis("off")
            
            axes[1].imshow(label_np, cmap="gray")
            axes[1].set_title("Ground Truth")
            axes[1].axis("off")
            
            pos = axes[2].imshow(prob_map, cmap="hot", vmin=0, vmax=1)
            axes[2].set_title("Attention U-Net Probability Map")
            axes[2].axis("off")
            fig.colorbar(pos, ax=axes[2], fraction=0.046, pad=0.04)
            
            axes[3].imshow(pred_mask, cmap="gray")
            axes[3].set_title("Predicted Binary Mask")
            axes[3].axis("off")
            
            plt.tight_layout()
            
            save_path = artifact_dir / f"val_pred_{img_id}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved visualization to {save_path}")

if __name__ == "__main__":
    main()
