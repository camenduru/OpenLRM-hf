import gradio as gr
import os
import subprocess
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

def assert_input_image(input_image):
    if input_image is None:
        raise gr.Error("No image selected or uploaded!")

def rembg_and_center_wrapper(source_image):
    subprocess.run([f'python rembg_and_center.py {source_image}'], shell=True)
    directory, filename = os.path.split(source_image)
    file_base, file_extension = os.path.splitext(filename)
    new_filename = f"{file_base}_rgba.png"
    new_image_path = os.path.join(directory, new_filename)
    return new_image_path

def infer_wrapper(source_image, checkbox_rembg):
    if checkbox_rembg:
        source_image = rembg_and_center_wrapper(source_image)
    return inferrer.infer(
        source_image=source_image,
        dump_path="./dumps",
        source_size=-1,
        render_size=-1,
        mesh_size=384,
        export_video=True,
        export_mesh=False,
    )

def infer_wrapper_clean(source_image):
    return infer_wrapper(source_image, checkbox_rembg=False)

def demo_image_to_video(inferrer: LRMInferrer):

    print(f"So far so good.")
    print(inferrer)

    _TITLE = '''OpenLRM: Open-Source Large Reconstruction Models'''

    _DESCRIPTION = '''
        <div>
            <a style="display:inline-block; href='https://github.com/3DTopia/OpenLRM'><img src='https://img.shields.io/github/stars/3DTopia/OpenLRM?style=social'/></a>
            <a style="display:inline-block; margin-left: .5em" href="https://huggingface.co/zxhezexin/OpenLRM"><img src='https://img.shields.io/badge/Model-Weights-blue'/></a>
            <a style="display:inline-block; margin-left: .5em" href="https://arxiv.org/abs/2311.04400"><img src="https://img.shields.io/badge/2311.04400-f9f7f7?logo=data:image/png;base64"></a>
        </div>
        OpenLRM is an open-source implementation of Large Reconstruction Models.
    '''

    with gr.Blocks(analytics_enabled=False) as iface:
        with gr.Row():

            with gr.Column(variant='panel'):
                with gr.Tabs(elem_id="openlrm_input_image"):
                    with gr.TabItem('Input Image'):
                        with gr.Row():
                            input_image = gr.Image(label="Input Image", image_mode="RGBA", sources="upload", type="filepath", elem_id="content_image", width="40%")

                with gr.Tabs(elem_id="openlrm_attrs"):
                    with gr.TabItem('Settings'):
                        with gr.Column(variant='panel'):
                            checkbox_rembg = gr.Checkbox(False,
                                             label='Remove background automatically')
                            submit = gr.Button('Generate', elem_id="openlrm_generate", variant='primary')

            with gr.Column(variant='panel'):
                with gr.Tabs(elem_id="openlrm_render_video"):
                    with gr.TabItem('Rendered Video'):
                        with gr.Row():
                            output_video = gr.Video(label="Rendered Video", format="mp4", width="80%")

        submit.click(
            fn=assert_input_image,
            inputs=[input_image],
            queue=False
        ).success(
            fn=infer_wrapper,
            inputs=[input_image, checkbox_rembg],
            outputs=[output_video],
        )

        with gr.Row():
            examples = [
                ['assets/sample_input/owl.png'],
                ['assets/sample_input/building.png'],
                ['assets/sample_input/mailbox.png'],
                ['assets/sample_input/fire.png'],
                ['assets/sample_input/girl.png'],
                ['assets/sample_input/lamp.png'],
                ['assets/sample_input/hydrant.png'],
                ['assets/sample_input/hotdogs.png'],
                ['assets/sample_input/traffic.png'],
                ['assets/sample_input/ceramic.png'],
            ]
            gr.Examples(
                examples=examples,
                inputs=[input_image], 
                outputs=[output_video],
                fn=infer_wrapper_clean,
                cache_examples=os.getenv('SYSTEM') == 'spaces',
            )
            
    return iface

if __name__ == "__main__":

    model_name = "lrm-base-obj-v1"

    prepare_checkpoint(model_name)

    with LRMInferrer(model_name) as inferrer:
        iface = demo_image_to_video(inferrer)
        iface.queue(max_size=10)
        iface.launch()
