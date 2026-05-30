# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from skimage.filters import frangi, threshold_otsu
from skimage.morphology import closing, opening, disk
from skimage.exposure import equalize_adapthist
from sklearn.metrics import jaccard_score, f1_score, accuracy_score, confusion_matrix

DATA_DIR = Path.cwd() / "data"
ORIGINAL_DIR = DATA_DIR / "original"
LABELS_DIR = DATA_DIR / "labels"


# %%
def split_channels(image: np.ndarray):
    """Return R, G, B channels from RGB image."""
    return image[:, :, 0], image[:, :, 1], image[:, :, 2]


# %%
def load_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path))


def load_label(path: Path) -> np.ndarray:
    lbl = np.array(Image.open(path))
    return (lbl > 0).astype(np.uint8)


# %%
def create_foreground_mask(image: np.ndarray, threshold: int = 30) -> np.ndarray:
    """Binary mask: True where sum(RGB) > threshold (eye region)."""
    return np.sum(image, axis=2) > threshold


def apply_mask(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero out pixels outside mask."""
    return data * mask


# %%
def enhance_green_channel(
    green: np.ndarray,
    mask: np.ndarray,
    clip_limit: float = 0.03,
    nbins: int = 256,
) -> np.ndarray:
    """Apply CLAHE to green channel within foreground mask."""
    normalized = green.astype(float) / 255.0
    normalized[~mask] = 0
    enhanced = equalize_adapthist(normalized, kernel_size=None, clip_limit=clip_limit, nbins=nbins)
    enhanced[~mask] = 0
    return enhanced


# %%
def frangi_vesselness(
    image: np.ndarray,
    sigmas: tuple = (1, 2, 3),
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: float = 15.0,
    black_ridges: bool = True,
) -> np.ndarray:
    """Apply Frangi vesselness filter. Returns vessel probability map."""
    return frangi(image, sigmas=sigmas, alpha=alpha, beta=beta, gamma=gamma, black_ridges=black_ridges)


# %%
def extract_veins(
    image: np.ndarray,
    sigmas: tuple = (1, 2, 3),
    clip_limit: float = 0.03,
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: float = 15.0,
) -> np.ndarray:
    """Extract veins from eye image. Returns binary vein mask."""
    _, green, _ = split_channels(image)
    fg_mask = create_foreground_mask(image)

    enhanced = enhance_green_channel(green, fg_mask, clip_limit=clip_limit)

    vesselness = frangi_vesselness(enhanced, sigmas=sigmas, alpha=alpha, beta=beta, gamma=gamma)
    vesselness_masked = apply_mask(vesselness, fg_mask)

    thresh = threshold_otsu(vesselness_masked[fg_mask])
    binary = vesselness_masked > thresh

    binary = closing(binary, disk(2))
    binary = opening(binary, disk(1))

    return binary.astype(np.uint8)


# %%
def evaluate(pred: np.ndarray, gold: np.ndarray) -> dict:
    """Compute evaluation metrics between prediction and gold standard."""
    p_flat = pred.ravel()
    g_flat = gold.ravel()

    return {
        "accuracy": accuracy_score(g_flat, p_flat),
        "f1": f1_score(g_flat, p_flat, zero_division=0),
        "iou": jaccard_score(g_flat, p_flat, zero_division=0),
        "tn": int(confusion_matrix(g_flat, p_flat).ravel()[0]),
        "fp": int(confusion_matrix(g_flat, p_flat).ravel()[1]),
        "fn": int(confusion_matrix(g_flat, p_flat).ravel()[2]),
        "tp": int(confusion_matrix(g_flat, p_flat).ravel()[3]),
    }


# %%
def visualize(
    image: np.ndarray,
    gold: np.ndarray,
    pred: np.ndarray,
    vesselness: np.ndarray,
    enhanced: np.ndarray,
    title: str = "",
):
    """Plot original, gold standard, enhanced, vesselness, and predicted veins."""
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    fig.suptitle(title, fontsize=13)

    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(gold, cmap="gray")
    axes[1].set_title("Gold Standard")
    axes[1].axis("off")

    axes[2].imshow(enhanced, cmap="gray")
    axes[2].set_title("CLAHE Enhanced")
    axes[2].axis("off")

    axes[3].imshow(vesselness, cmap="hot")
    axes[3].set_title("Frangi Vesselness")
    axes[3].axis("off")

    axes[4].imshow(pred, cmap="gray")
    axes[4].set_title("Predicted Veins")
    axes[4].axis("off")

    plt.tight_layout()
    plt.show()


# %%
def process_one(image_id: str) -> dict:
    """Load, extract veins, evaluate for a single image."""
    img = load_image(ORIGINAL_DIR / f"{image_id}.ppm")
    gold = load_label(LABELS_DIR / f"{image_id}.vk.ppm")

    _, green, _ = split_channels(img)
    fg_mask = create_foreground_mask(img)

    enhanced = enhance_green_channel(green, fg_mask)

    vesselness = frangi_vesselness(enhanced, black_ridges=True)
    vesselness_masked = apply_mask(vesselness, fg_mask)

    thresh = threshold_otsu(vesselness_masked[fg_mask])
    binary = vesselness_masked > thresh
    binary = closing(binary, disk(2))
    binary = opening(binary, disk(1))
    pred = binary.astype(np.uint8)

    metrics = evaluate(pred, gold)
    return {
        "image_id": image_id,
        "image": img,
        "gold": gold,
        "pred": pred,
        "vesselness": vesselness,
        "enhanced": enhanced,
        **metrics,
    }


# %%
image_ids = sorted([p.stem for p in ORIGINAL_DIR.glob("*.ppm")])
image_ids = list(image_ids)[:1]
print(f"Found {len(image_ids)} images: {image_ids[:5]}...")

# %%
results = []
for img_id in image_ids:
    r = process_one(img_id)
    results.append(r)
    print(f"{img_id}: acc={r['accuracy']:.3f}  f1={r['f1']:.3f}  iou={r['iou']:.3f}")

# %%
df = pd.DataFrame(results)
df[["image_id", "accuracy", "f1", "iou", "tp", "fp", "fn", "tn"]]

# %%
print(f"Mean accuracy: {df['accuracy'].mean():.3f}")
print(f"Mean F1:       {df['f1'].mean():.3f}")
print(f"Mean IoU:      {df['iou'].mean():.3f}")

# %%
best_idx = df["f1"].idxmax()
worst_idx = df["f1"].idxmin()

visualize(
    results[best_idx]["image"],
    results[best_idx]["gold"],
    results[best_idx]["pred"],
    results[best_idx]["vesselness"],
    results[best_idx]["enhanced"],
    title=f"Best: {results[best_idx]['image_id']}  F1={results[best_idx]['f1']:.3f}",
)

# %%
visualize(
    results[worst_idx]["image"],
    results[worst_idx]["gold"],
    results[worst_idx]["pred"],
    results[worst_idx]["vesselness"],
    results[worst_idx]["enhanced"],
    title=f"Worst: {results[worst_idx]['image_id']}  F1={results[worst_idx]['f1']:.3f}",
)
