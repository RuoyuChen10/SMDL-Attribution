# build_imagenet_clip_text_prototypes.py
import os, torch, clip
from utils import imagenet_classes, imagenet_templates

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load("ViT-L/14", device=device)

# 可调：为加速可只取前 7 个模板
templates = imagenet_templates[:7]

texts = []
for cls in imagenet_classes:
    prompts = [t.format(cls.replace("_"," ")) for t in templates]
    texts += prompts

with torch.no_grad():
    tokens = clip.tokenize(texts).to(device)
    text_features = model.encode_text(tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

tpn = len(templates)
per_class = []
for i in range(len(imagenet_classes)):
    ft = text_features[i*tpn:(i+1)*tpn].mean(0)
    ft = ft / ft.norm()
    per_class.append(ft)
semantic_feature = torch.stack(per_class, 0).to(device)

save_path = "ckpt/semantic_features/clip_vitl_imagenet_zeroweights.pt"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(semantic_feature.detach().cpu(), save_path)
print("Saved:", save_path, "shape:", semantic_feature.shape)
