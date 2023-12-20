import gradio as gr
from huggingface_hub import hf_hub_download

from lrm.inferrer import LRMInferrer

def prepare_checkpoint(model_name: str):

    REPO_ID = f"zxhezexin/OpenLRM"
    FILE_NAME = f"{model_name}.pth"
    CACHE_PATH = f".cache"

    print(f"Downloading ckpt ...")

    ckpt_path = hf_hub_download(repo_id=REPO_ID, filename=FILE_NAME, local_dir=CACHE_PATH)
    print(f"checkpoint path is {ckpt_path}")

    print(f"Downloaded ckpt into {CACHE_PATH}")


def demo_image_to_video(inferrer: LRMInferrer):

    print(f"So far so good.")
    print(inferrer)

    with gr.Blocks(analytics_enabled=False) as iface:
        with gr.Row():

            with gr.Column(variant='panel'):
                with gr.Tabs(elem_id="openlrm_input_image"):
                    with gr.TabItem('Input Image'):
                        with gr.Row():
                            input_image = gr.Image(label="Input Image", sources="upload", type="filepath", elem_id="content_image", width=512)

                with gr.Tabs(elem_id="openlrm_attrs"):
                    with gr.TabItem('Settings'):
                        with gr.Column(variant='panel'):
                            submit = gr.Button('Generate', elem_id="sadtalker_generate", variant='primary')

                with gr.Tabs(elem_id="openlrm_render_video"):
                    gen_video = gr.Video(label="Rendered Video", format="mov", width=512)

        submit.click(
            fn=inferrer.infer,
            inputs={
                "source_image": input_image,
                "export_video": True,
            }, 
            outputs=[gen_video]
            )
    return iface

if __name__ == "__main__":

    model_name = "lrm-base-obj-v1"

    prepare_checkpoint(model_name)

    with LRMInferrer(model_name) as inferrer:
        iface = demo_image_to_video(inferrer)
        iface.queue(max_size=10)
        iface.launch(debug=True)
