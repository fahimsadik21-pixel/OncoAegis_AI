import os
import numpy as np
import matplotlib.pyplot as plt


# ==============================
# Dataset Path
# ==============================

DATA_DIR = r"U:\OncoAegis_AI\datasets\processed\luna16"


# ==============================
# Find CT file
# ==============================

ct_files = sorted(
    [
        f for f in os.listdir(DATA_DIR)
        if f.endswith("_ct.npy")
    ]
)


if len(ct_files) == 0:
    raise FileNotFoundError(
        "No CT .npy file found!"
    )


ct_file = ct_files[0]

mask_file = ct_file.replace(
    "_ct.npy",
    "_mask.npy"
)


ct_path = os.path.join(
    DATA_DIR,
    ct_file
)

mask_path = os.path.join(
    DATA_DIR,
    mask_file
)


print("CT File:")
print(ct_file)

print("\nMask File:")
print(mask_file)



# ==============================
# Load numpy data
# ==============================

ct = np.load(ct_path)

mask = np.load(mask_path)


print("\nCT Shape:")
print(ct.shape)

print("\nMask Shape:")
print(mask.shape)



# ==============================
# Select middle slice
# ==============================

slice_id = ct.shape[0] // 2


ct_slice = ct[slice_id]

mask_slice = mask[slice_id]


print("\nViewing slice:")
print(slice_id)



# ==============================
# Visualization
# ==============================

plt.figure(
    figsize=(12, 5)
)


# CT image

plt.subplot(
    1,
    2,
    1
)

plt.imshow(
    ct_slice,
    cmap="gray"
)

plt.title(
    "LUNA16 CT Slice"
)

plt.axis(
    "off"
)



# CT + Mask overlay

plt.subplot(
    1,
    2,
    2
)


plt.imshow(
    ct_slice,
    cmap="gray"
)


plt.imshow(
    mask_slice,
    cmap="jet",
    alpha=0.35
)


plt.title(
    "CT + Lung Segmentation Mask"
)

plt.axis(
    "off"
)



plt.tight_layout()


plt.show()