import gradio as gr
import os
import shutil
from huggingface_hub import hf_hub_download

from lrm.inferrer import LRMInferrer

def prepare_checkpoint(model_name: str):

    REPO_ID = f"zxhezexin/OpenLRM"
    FILE_NAME = f"{model_name}.pth"
    REVISION = f"main"
    CACHE_PATH = f".cache"

    ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
    os.makedirs(CACHE_PATH, exist_ok=True)
    shutil.move(ckpt_path, os.path.join(CACHE_PATH, f"{FILE_NAME}"))

    print(f"Downloaded ckpt into {CACHE_PATH}")


def demo_image_to_video(inferrer: LRMInferrer):

    return "Hello " + name + "!!"

if __name__ == "__main__":

    model_name = "lrm-base-obj-v1"

    prepare_checkpoint

    with LRMInferrer(model_name) as inferrer:
        iface = demo_image_to_video(inferrer)
        iface = gr.Interface(fn=greet, inputs="text", outputs="text")
        iface.launch()
