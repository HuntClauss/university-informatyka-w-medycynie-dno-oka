import json

notebook_path = "/home/redsec/.djozwik2/university-informatyka-w-medycynie-dno-oka/unet.ipynb"
with open(notebook_path, 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            if 'orig_img_path = ORIGINAL_DIR' in line:
                source[i] = "        orig_img = images[i].permute(1, 2, 0).cpu().numpy()\n"
            elif 'orig_img = Image.open(' in line:
                source[i] = "        if orig_img.max() > 1.0:\n            orig_img = orig_img / 255.0\n"
            elif 'axes[0].set_title("Original Image")' in line or "axes[0].set_title('Original Image')" in line:
                source[i] = "        axes[0].set_title(\"Preprocessed Input Image\")\n"
        cell['source'] = source

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)
