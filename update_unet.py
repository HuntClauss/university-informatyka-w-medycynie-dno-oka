import json
import re

notebook_path = "/home/redsec/.djozwik2/university-informatyka-w-medycynie-dno-oka/unet.ipynb"
with open(notebook_path, 'r') as f:
    nb = json.load(f)

# 1. Update BCEDiceLoss in cell 10
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'class BCEDiceLoss(nn.Module):' in "".join(cell['source']):
        source = "".join(cell['source'])
        new_source = source.replace(
            "def __init__(self, bce_weight=0.5, dice_weight=0.5):",
            "def __init__(self, bce_weight=0.5, dice_weight=0.5, pos_weight=None):"
        )
        new_source = new_source.replace(
            "self.bce = nn.BCEWithLogitsLoss()",
            "self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)"
        )
        cell['source'] = [line + "\n" if not line.endswith("\n") else line for line in new_source.split("\n")[:-1]]

# 2. Update training loop in cell 12
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'model = UNet' in "".join(cell['source']):
        source = "".join(cell['source'])
        new_source = source.replace(
            "criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)",
            "pos_weight = torch.tensor([5.0]).to(device)\n    criterion = BCEDiceLoss(bce_weight=0.3, dice_weight=0.7, pos_weight=pos_weight)"
        )
        new_source = new_source.replace(
            "print(f\"Epoch {epoch:02d}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | Val IoU: {val_metrics['iou']:.4f} | Val F1: {val_metrics['f1']:.4f}\")",
            "print(f\"Epoch {epoch:02d}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | Val IoU: {val_metrics['iou']:.4f} | Val F1: {val_metrics['f1']:.4f} | Sen: {val_metrics['sensitivity']:.4f} | Spe: {val_metrics['specificity']:.4f}\")"
        )
        # Fix indentation for pos_weight
        new_source = new_source.replace("    pos_weight", "pos_weight")
        cell['source'] = [line + "\n" if not line.endswith("\n") else line for line in new_source.split("\n")[:-1]]
        # Clear outputs so the user can re-run
        cell['outputs'] = []

# 3. Update validate_epoch and plot_predictions threshold and image display
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'def validate_epoch' in "".join(cell['source']):
        source = "".join(cell['source'])
        new_source = source.replace(
            "preds = (probs > 0.5).float()",
            "preds = (probs > 0.3).float()"
        )
        cell['source'] = [line + "\n" if not line.endswith("\n") else line for line in new_source.split("\n")[:-1]]

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'def plot_predictions' in "".join(cell['source']):
        source = "".join(cell['source'])
        
        # We need to change how the original image is plotted.
        # Currently:
        # orig_img_path = ORIGINAL_DIR / f"{img_id}.ppm"
        # orig_img = Image.open(orig_img_path)
        # axes[0].imshow(orig_img)
        # We change it to use the preprocessed image tensor (which is `images[i]`)
        
        new_source = source.replace(
            "orig_img_path = ORIGINAL_DIR / f\"{img_id}.ppm\"\n        orig_img = Image.open(orig_img_path)",
            "orig_img = images[i].permute(1, 2, 0).cpu().numpy()\n        # Normalize back to 0-1 for visualization if needed\n        if orig_img.max() > 1.0:\n            orig_img = orig_img / 255.0"
        )
        new_source = new_source.replace(
            "axes[0].set_title(\"Original Image\")",
            "axes[0].set_title(\"Preprocessed Input Image\")"
        )
        cell['source'] = [line + "\n" if not line.endswith("\n") else line for line in new_source.split("\n")[:-1]]
        cell['outputs'] = []

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)
