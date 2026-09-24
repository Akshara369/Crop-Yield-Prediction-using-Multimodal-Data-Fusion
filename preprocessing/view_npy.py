import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NPY = ROOT / "datasets" / "test_images" / "punjab_2020_rice_env_tensor.npy"


def view_npy_file(npy_path: Path | str) -> None:
    npy_path = Path(npy_path)

    if not npy_path.exists():
        print(f"[!] File not found: {npy_path}")
        return

    print(f"\n============================================================")
    print(f"INSPECTING NUMPY ARRAY: {npy_path.name}")
    print(f"============================================================")

    # 1. Load array
    arr = np.load(npy_path)
    print(f"[+] Array Shape    : {arr.shape}")
    print(f"[+] Data Type      : {arr.dtype}")
    print(f"[+] Value Range    : Min = {arr.min():.4f}, Max = {arr.max():.4f}")

    # 2. If 3D array (Height, Width, Channels), plot each channel
    if arr.ndim == 3:
        num_channels = arr.shape[-1]
        print(f"[+] Displaying {num_channels} channels...")

        fig, axes = plt.subplots(1, num_channels, figsize=(4 * num_channels, 4))
        if num_channels == 1:
            axes = [axes]

        channel_names = ["Rainfall (mm)", "Max Temp (°C)", "Min Temp (°C)", "Soil Organic Carbon"]
        cmaps = ["Blues", "YlOrRd", "Oranges", "YlGn"]

        for i in range(num_channels):
            c_name = channel_names[i] if i < len(channel_names) else f"Channel {i+1}"
            cmap = cmaps[i] if i < len(cmaps) else "viridis"

            im = axes[i].imshow(arr[:, :, i], cmap=cmap)
            axes[i].set_title(f"{c_name}\n({arr[:, :, i].shape[0]}x{arr[:, :, i].shape[1]})", fontsize=10, fontweight="bold")
            axes[i].axis("off")
            plt.colorbar(im, ax=axes[i], shrink=0.7)

        plt.tight_layout()
        preview_out = npy_path.parent / f"preview_{npy_path.stem}.png"
        plt.savefig(preview_out, dpi=200, bbox_inches="tight")
        print(f"[+] Saved visual preview PNG to: {preview_out}")
        try:
            plt.show()
        except Exception:
            pass

    elif arr.ndim == 2:
        plt.figure(figsize=(6, 5))
        plt.imshow(arr, cmap="viridis")
        plt.title(f"{npy_path.stem} ({arr.shape[0]}x{arr.shape[1]})", fontweight="bold")
        plt.colorbar()
        plt.axis("off")
        plt.show()
    else:
        print(f"[!] Array has {arr.ndim} dimensions. Reshape or inspect directly via print(arr).")


if __name__ == "__main__":
    filepath = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_NPY
    view_npy_file(filepath)
