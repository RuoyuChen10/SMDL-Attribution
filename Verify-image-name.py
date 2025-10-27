# 快检
import os
root=r"datasets\imagenet\ILSVRC2012_img_val"
for i,l in enumerate(open("datasets/imagenet/val_clip_vitl_10.txt","r",encoding="utf-8"),1):
    rel=l.strip().rsplit(" ",1)[0]
    p=os.path.join(root,rel)
    if not os.path.exists(p): print("missing:", i, rel)