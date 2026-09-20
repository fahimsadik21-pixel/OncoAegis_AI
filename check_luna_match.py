from pathlib import Path

base = Path(r"U:\OncoAegis_AI\datasets\universal_ct\luna16")

seg_dir = base / "seg-lungs-LUNA16"
ct_dir = base / "subset1"

# segmentation IDs
seg_ids = {
    f.stem for f in seg_dir.glob("*.mhd")
}

# CT IDs
ct_ids = {
    f.stem for f in ct_dir.glob("*.mhd")
}

matched = seg_ids.intersection(ct_ids)

print("Segmentation files:", len(seg_ids))
print("CT files:", len(ct_ids))
print("Matched pairs:", len(matched))
print("Missing CT:", len(seg_ids - ct_ids))
print("Extra CT:", len(ct_ids - seg_ids))