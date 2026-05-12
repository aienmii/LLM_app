import os
import gradio as gr
from huggingface_hub import InferenceClient


def inference(promt, token):
    return "a","b","c"

def hide():
    return gr.Textbox(visible="hidden")

with gr.Blocks() as demo:
    gr.Markdown("<center><h1>AIHub</h1></center>")

    promt = gr.Textbox(label="Input prompt")
    token = gr.Textbox(label="Token")

    gen_button = gr.Button(
        "Run",
        variant="primary"
    )

    with gr.Row():
        llama_output = gr.Markdown("Llama model")
        deepseek_output = gr.Markdown("DeepSeek model")
        nous_output = gr.Markdown("Nous Hermes model")


    gr.on([promt.submit,gen_button.click],
        fn= hide,
        inputs=None,
        outputs=[token]
          ).then(
        fn=inference,
        nputs=[promt, token],
        outputs=[llama_output, deepseek_output, nous_output],


    )



if __name__ == "__main__":
    demo.launch()