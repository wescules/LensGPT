#!/usr/bin/env python3
"""
    MediaSearch – semantic search for local photos & videos using CLIP + FAISS.
    Usage:
        python main.py index '/path/to/media'
        python main.py search 
"""
import sys
from MediaSeach import MediaSearch
from GradioUI import GradioUI

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "index" and len(sys.argv) == 3:
        folder = sys.argv[2]
        print("Building Index")
        MediaSearch().build_index(folder)
    elif cmd == "search":
        GradioUI().display_UI()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
