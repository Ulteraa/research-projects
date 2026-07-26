from pathlib import Path
import argparse
from PIL import Image
VALID={".jpg",".jpeg",".png",".tif",".tiff",".webp"}
p=argparse.ArgumentParser(); p.add_argument("--scene_dir",type=Path,required=True); a=p.parse_args()
image_dir=a.scene_dir/"images"
if not image_dir.is_dir(): raise FileNotFoundError(image_dir)
paths=sorted(x for x in image_dir.iterdir() if x.suffix.lower() in VALID)
if len(paths)<2: raise RuntimeError(f"Need at least 2 images; found {len(paths)}")
sizes={}
for path in paths:
    with Image.open(path) as im:
        im.verify()
    with Image.open(path) as im:
        sizes[im.size]=sizes.get(im.size,0)+1
print(f"Images: {len(paths)}")
for size,count in sizes.items(): print(f"{size[0]}x{size[1]}: {count}")
print("Scene validation passed.")
