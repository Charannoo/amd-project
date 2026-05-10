# Hugging Face Space Entrypoint
import os

# Ensure we don't try to create a share link on Hugging Face Spaces
os.environ["GRADIO_SHARE"] = "False"

# Import the demo from the UI folder
from ui.app import demo

if __name__ == "__main__":
    demo.launch()
