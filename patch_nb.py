import json

with open("ml.ipynb", "r") as f:
    nb = json.load(f)

# Cell 0 imports update
# add `from skimage.util import view_as_windows`
cell0_source = nb["cells"][0]["source"]
import_str = "from skimage.util import view_as_windows\n"
if import_str not in "".join(cell0_source):
    nb["cells"][0]["source"].insert(9, import_str)

# Cell 3 function update
new_cell3 = """def extract_features(
    img_rgb: np.ndarray, 
    mask: np.ndarray,
    *,
    sigmas_gauss: list[float] = [1, 2, 4],
    sigmas_hessian: list[float] = [1, 2],
    sigmas_vessel: tuple[float] = (0.5, 1, 1.5, 2, 2.5, 3),
    N: int = 3
) -> np.ndarray:
    _, green, _ = split_channels(img_rgb)
    enhanced = preprocess_image(green, mask)
    
    feats = [enhanced]
    
    for sigma in sigmas_gauss:
        blur = cv2.GaussianBlur(enhanced, (0, 0), sigma)
        feats.append(blur)
        
    feats.append(sobel(enhanced))
    
    for sigma in sigmas_hessian:
        H = hessian_matrix(enhanced, sigma=sigma, order='rc')
        eigvals = hessian_matrix_eigvals(H)
        feats.append(eigvals[0])
        feats.append(eigvals[1])
        
    v_frangi = frangi(enhanced, sigmas=sigmas_vessel, alpha=0.5, beta=0.5, gamma=None, black_ridges=False)
    feats.append(v_frangi)
    
    v_meijering = meijering(enhanced, sigmas=sigmas_vessel, black_ridges=False)
    feats.append(v_meijering)
    
    v_sato = sato(enhanced, sigmas=sigmas_vessel, black_ridges=False)
    feats.append(v_sato)
    
    pixel_feats = np.stack(feats, axis=-1)
    
    if N == 1:
        return pixel_feats
        
    pad_size = N // 2
    # Pad spatial dimensions with reflection
    padded = np.pad(pixel_feats, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='reflect')
    
    # Extract patches using sliding window
    # Window shape: N, N, num_features
    windows = view_as_windows(padded, (N, N, pixel_feats.shape[-1]))
    # Squeeze the 1-dim at axis 2 (channel output from view_as_windows)
    windows = windows.squeeze(axis=2)
    
    H, W = pixel_feats.shape[:2]
    num_features = pixel_feats.shape[-1]
    
    # Reshape to (H, W, N*N*num_features)
    return windows.reshape(H, W, N * N * num_features)"""

for cell in nb["cells"]:
    if cell["cell_type"] == "code" and "def extract_features" in "".join(cell["source"]):
        lines = [line + "\n" for line in new_cell3.split("\n")]
        lines[-1] = lines[-1].rstrip("\n")
        cell["source"] = lines

with open("ml.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Notebook updated.")
