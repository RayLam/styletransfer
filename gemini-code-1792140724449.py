import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def export_all_patches_to_file(
    image_path: str, output_txt: str = "all_patches_full_dump.txt"
):
    # 1. Load image and normalize [0.0, 1.0]
    pil_img = Image.open(image_path).convert("RGB")
    hwc_array = np.array(pil_img, dtype=np.float32) / 255.0
    raw_H, raw_W, C = hwc_array.shape

    # 2. Convert to PyTorch Batch Tensor [1, C, H, W]
    tensor_batch = (
        torch.from_numpy(hwc_array).permute(2, 0, 1).unsqueeze(0)
    )

    # 3. Auto-Pad to ensure H and W are divisible by 16
    pad_h = (16 - (raw_H % 16)) % 16
    pad_w = (16 - (raw_W % 16)) % 16
    padded_batch = F.pad(
        tensor_batch, (0, pad_w, 0, pad_h), mode="reflect"
    )
    _, _, H, W = padded_batch.shape

    n_patches_h = H // 16
    n_patches_w = W // 16
    total_patches = n_patches_h * n_patches_w

    # 4. Extract and flatten 16x16x3 -> 768
    patches = padded_batch.unfold(2, 16, 16).unfold(3, 16, 16)
    patches = patches.permute(0, 2, 3, 4, 5, 1)
    patch_sequence = patches.reshape(total_patches, 16 * 16 * C).numpy()

    print(
        f"Exporting ALL {total_patches} patches and all 768 values per patch to '{output_txt}'..."
    )

    # 5. Write EVERY SINGLE value to file
    with open(output_txt, "w") as f:
        f.write(
            f"=== FULL PATCH DATA DUMP FOR IMAGE: {image_path} ===\n"
        )
        f.write(
            f"Grid Layout: {n_patches_h} rows x {n_patches_w} cols | Total Patches: {total_patches}\n"
        )
        f.write(f"Tensor Shape: [{total_patches}, 768]\n\n")

        for idx in range(total_patches):
            r = idx // n_patches_w
            c = idx % n_patches_w
            vec = patch_sequence[idx]

            f.write(f"==================================================\n")
            f.write(
                f"PATCH INDEX: {idx} | GRID: (Row {r}, Col {c}) | Y:[{r*16}:{(r+1)*16}], X:[{c*16}:{(c+1)*16}]\n"
            )
            f.write(f"--------------------------------------------------\n")

            # Format the 768 float values nicely in rows of 12 numbers
            formatted_vals = ", ".join([f"{val:.4f}" for val in vec])
            f.write(f"VALUES (768 Entries):\n[{formatted_vals}]\n\n")

    print(f"SUCCESS! Open '{output_txt}' to inspect every single value.")


# Execute the export
# export_all_patches_to_file("image_d79821.png")