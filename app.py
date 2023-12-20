import gradio as gr

from lrm.inferrer import LRMInferrer

def demo_image_to_video(name):
    return "Hello " + name + "!!"

iface = gr.Interface(fn=greet, inputs="text", outputs="text")
iface.launch()
