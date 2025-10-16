#!/usr/bin/env python3
"""
MediaSearch – semantic search for local photos & videos using CLIP + FAISS.
Usage:
    python mediasearch.py index /path/to/media
    python mediasearch.py search "sunset on the beach"
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import faiss
import pickle
from tqdm import tqdm
import cv2
from transformers import CLIPProcessor, CLIPModel

# ---------------- CONFIG ---------------- #
INDEX_FILE = "media.index"
META_FILE = "media_meta.pkl"
FRAME_INTERVAL = 5     # seconds between sampled frames in videos
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------------------------- #

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def get_image_embedding(image: Image.Image):
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
    emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    return emb.cpu().numpy().squeeze()


def extract_keyframes(video_path, every_n_seconds=FRAME_INTERVAL):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30
    frame_interval = int(fps * every_n_seconds)
    frames = []
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        i += 1
    cap.release()
    return frames


def build_index(folder):
    paths = []
    embeddings = []

    files = [p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS]
    print(f"Found {len(files)} media files")

    for p in tqdm(files, desc="Indexing"):
        try:
            if p.suffix.lower() in IMAGE_EXTS:
                img = Image.open(p).convert("RGB")
                emb = get_image_embedding(img)
            else:
                frames = extract_keyframes(p)
                if not frames:
                    continue
                embs = np.stack([get_image_embedding(f) for f in frames])
                emb = embs.mean(axis=0)
            embeddings.append(emb)
            paths.append(str(p))
        except Exception as e:
            print(f"Error {p}: {e}")

    if not embeddings:
        print("No embeddings generated.")
        return

    embeddings = np.stack(embeddings).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "wb") as f:
        pickle.dump(paths, f)

    print(f"✅ Indexed {len(paths)} items. Saved to {INDEX_FILE} and {META_FILE}.")


def search(query, top_k=10):
    if not (os.path.exists(INDEX_FILE) and os.path.exists(META_FILE)):
        print("❌ No index found. Run `index` first.")
        return

    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "rb") as f:
        paths = pickle.load(f)

    inputs = processor(text=[query], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        text_emb = model.get_text_features(**inputs)
    text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)
    text_emb = text_emb.cpu().numpy().astype("float32")

    scores, idxs = index.search(text_emb, top_k)
    print(f"\n🔍 Results for: “{query}”\n")
    for rank, (score, idx) in enumerate(zip(scores[0], idxs[0]), 1):
        print(f"{rank:2d}. {paths[idx]}  (score: {score:.3f})")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "index" and len(sys.argv) == 3:
        folder = sys.argv[2]
        print("Building Index")
        build_index(folder)
    elif cmd == "search" and len(sys.argv) >= 3:
        query = " ".join(sys.argv[2:])
        search(query)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
