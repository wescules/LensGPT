# Pixplore — Semantic Media Search

Pixplore is a personal media search tool that lets you search through your photos and videos using natural language. It uses OpenAI’s CLIP model to understand the content of images and videos, so you can find what you’re looking for without needing to remember file names or folders.

---

## Features

* Search **all your images and videos** using plain English.
* Supports **large media collections** (tested on 70GB+ libraries).
* Shows **image thumbnails** and **video previews** in a gallery.
* Click on a video thumbnail to play it directly in the UI.
* “Show More” button to gradually load results without overwhelming your system.
* Fully **offline after initial setup** — your media never leaves your computer.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/Pixplore.git
cd Pixplore
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

> Make sure you have Python 3.11+ installed.


---

## Usage

1. Index your media:

```bash
python main.py index /path/to/your/media
```

This will create embeddings for all images and videos in the folder and save them locally.

2. Launch the UI:

```bash
python main.py search
```

<img width="1824" height="2000" alt="image" src="https://github.com/user-attachments/assets/319306cd-cabd-4d8f-8230-83713e7e4d2f" />

* Enter a search query in natural language.
* Browse the gallery of results.
* Click on images or video thumbnails to open/play them.

---

## Supported Media

* Images: `.jpg`, `.jpeg`, `.png`, `.webp`
* Videos: `.mp4`, `.mov`, `.avi`, `.mkv`

> Video thumbnails are automatically generated from the first frame.

---

## Notes

* The first run will download the CLIP model (~400 MB). After that, everything works offline.
* Designed to handle **medium-to-large collections**, but indexing can take time for tens of thousands of files.
---
