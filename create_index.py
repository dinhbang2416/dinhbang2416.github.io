import torch
import clip
from PIL import Image
import os
import numpy as np
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

IMG_DIR = r"C:\CLIP\archive (5)\flickr30k_images\flickr30k_images"
image_files = [f for f in os.listdir(IMG_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]

all_features = []
file_names = []

print(f"Đang xử lý {len(image_files)} ảnh...")

for filename in tqdm(image_files):
    img_path = os.path.join(IMG_DIR, filename)
    try:
        image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model.encode_image(image)
            features /= features.norm(dim=-1, keepdim=True)
            all_features.append(features.cpu().numpy())
            file_names.append(filename)
    except:
        continue

# Lưu vào file
np.save("image_features.npy", np.vstack(all_features))
np.save("file_names.npy", np.array(file_names))
print("Đã tạo xong bộ nhớ vector!")