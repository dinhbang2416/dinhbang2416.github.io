import streamlit as st
import torch
import clip
import numpy as np
import os
from PIL import Image

st.set_page_config(layout="wide")
st.title("⚡ Siêu công cụ tìm kiếm ảnh (30,000+ ảnh)")

@st.cache_resource
def load_data():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device)
    # Load bộ nhớ đã tạo từ bước 1
    features = np.load("image_features.npy")
    filenames = np.load("file_names.npy")
    return model, device, features, filenames

model, device, features_db, filenames_db = load_data()

query = st.text_input("Nhập nội dung cần tìm:", "two dogs playing in the snow")
top_k = st.slider("Số lượng kết quả", 4, 20, 8)

if query:
    # 1. Chuyển câu hỏi của người dùng thành vector
    text_tokens = clip.tokenize([query]).to(device)
    with torch.no_grad():
        text_feature = model.encode_text(text_tokens)
        text_feature /= text_feature.norm(dim=-1, keepdim=True)
        text_feature = text_feature.cpu().numpy()

    # 2. Tính toán độ tương đồng (Nhân ma trận - Cực nhanh!)
    similarities = (text_feature @ features_db.T)[0]
    
    # 3. Lấy Top K kết quả
    best_indices = similarities.argsort()[::-1][:top_k]

    # 4. Hiển thị kết quả
    cols = st.columns(4)
    IMG_DIR = r"C:\CLIP\archive (5)\flickr30k_images\flickr30k_images"
    
    for i, idx in enumerate(best_indices):
        with cols[i % 4]:
            fname = filenames_db[idx]
            score = similarities[idx]
            img_path = os.path.join(IMG_DIR, fname)
            st.image(Image.open(img_path), caption=f"Match: {score:.2f}")