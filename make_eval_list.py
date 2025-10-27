import os, argparse, random, sys

p = argparse.ArgumentParser()
p.add_argument("--datasets", required=True)
p.add_argument("--out", required=True)
p.add_argument("--num", type=int, default=10)      # 0 表示全部
p.add_argument("--class_id", type=int, default=207)
a = p.parse_args()

imgs = []
for r, _, fs in os.walk(a.datasets):
    for f in fs:
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            rel = os.path.relpath(os.path.join(r, f), a.datasets).replace("\\", "/")
            imgs.append(rel)

if not imgs:
    print("No images found under", a.datasets)
    sys.exit(2)

random.shuffle(imgs)
sel = imgs if a.num <= 0 or a.num > len(imgs) else imgs[:a.num]

os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "w", encoding="utf-8") as w:
    for rel in sel:
        w.write(f"{rel} {a.class_id}\n")
print("Wrote", len(sel), "lines to", a.out)