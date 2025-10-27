import os, random

root = r"datasets/imagenet/ILSVRC2012_img_val"
files = [f for f in os.listdir(root) if f.lower().endswith(('.jpeg', '.jpg'))]
random.shuffle(files)
sel = files[:10]  # 随机取10张

os.makedirs(r"datasets/imagenet", exist_ok=True)
with open(r"datasets/imagenet/val_clip_vitl_10.txt", "w", encoding="utf-8") as f:
    for name in sel:
        f.write(f"{name} 207\n")

print("Wrote datasets/imagenet/val_clip_vitl_10.txt:", sel)