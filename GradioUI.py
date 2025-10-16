import gradio as gr
import os
import cv2
from PIL import Image
from matplotlib.pyplot import grid
from MediaSeach import MediaSearch
class GradioUI:
    THUMB_SIZE = (500, 500)  # thumbnail size for display

    def display_UI(self):
        with gr.Blocks() as demo:
            gr.Markdown("## Pixplore Media Search")
            
            with gr.Row():
                query_input = gr.Textbox(label="Search query")
                search_btn = gr.Button("Search")
            
            with gr.Row():
                gallery = gr.Gallery(label="Results", show_label=True, elem_classes="gallery", height=1600, columns=4)
            
            # Hidden file paths to handle clicks
            file_paths_state = gr.State()

            def on_search(query):
                thumbs, paths = self.display_search(query)
                return thumbs, paths

            search_btn.click(fn=on_search, inputs=query_input, outputs=[gallery, file_paths_state])

            # Open file on click
            def on_click(evt: gr.SelectData, paths):
                if paths and 0 <= evt.index < len(paths):
                    self.open_file(paths[evt.index])
            
            gallery.select(fn=on_click, inputs=[file_paths_state], outputs=[])

            demo.launch(pwa=True)

    def make_thumbnail(self, path):
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext in {".mp4", ".mov", ".avi", ".mkv"}:
                cap = cv2.VideoCapture(path)
                ret, frame = cap.read()
                cap.release()
                if not ret:
                    return None
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
            else:
                img = Image.open(path).convert("RGB")
            img.thumbnail(self.THUMB_SIZE)
            return img
        except Exception as e:
            print(f"Error creating thumbnail for {path}: {e}")
            return None

    def open_file(self, path):
        if os.name == 'nt':  # Windows
            os.startfile(path)
        else:
            print("Not supported")

    def display_search(self, query):
        scores, idxs, paths = MediaSearch().search(query, top_k=30)
        thumbs = []
        file_paths = []
        for rank, (score, idx) in enumerate(zip(scores[0], idxs[0]), 1):
            thumb = self.make_thumbnail(paths[idx])
            if thumb:
                thumbs.append(thumb)
                file_paths.append(paths[idx])
        return thumbs, file_paths
