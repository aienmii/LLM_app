import os
from functools import partial

import gradio as gr
from huggingface_hub import InferenceClient

system_promt = """
You are helpful,respectful asistant.Always answer as helpfullt as possibly,while being safe. 
"""


def inference(promt, hf_token, model, model_name):
    messages = [{"role": "system", "content": system_promt}, {"role": "user", "content": promt}]
    if hf_token is None or not hf_token.strip():
        hf_token = os.getenv("HF_TOKEN")
    client = InferenceClient(model=model, token=hf_token)
    tokens = f"**{model_name}**\n"

    for completion in client.chat_completion(messages, max_tokens=200, stream=True):
        token = completion.choices[0].delta.content
        if token is not None:
            tokens += token
            yield tokens


def hide():
    return gr.Textbox(visible="hidden")


with gr.Blocks() as demo:
    gr.Markdown("<center><h1>AIHub</h1></center>")

    promt = gr.Textbox(label="Input prompt")
    token = gr.Textbox(label="Token",
                       visible=False
)

    gen_button = gr.Button(
        "Run",
        variant="primary"
    )

    with gr.Row():
        llama_output = gr.Markdown("Llama-3.3-70B-Instruct model")
        mistral_output = gr.Markdown("<h1>gemma-4-26B-A4B-it<h1>")
        qwen_output = gr.Markdown("Qwen2.5-72B-Instruct model")

    gr.on(triggers=[promt.submit, gen_button.click],
          fn=hide,
          inputs=None,
          outputs=[token],
          show_progress="hidden"
          )
    gr.on(triggers=[promt.submit, gen_button.click],
          fn=partial(inference, model="meta-llama/Llama-3.3-70B-Instruct", model_name="Llama-3.3-70B-Instruct"),
          inputs=[promt, token],
          outputs=[llama_output],
          show_progress="hidden"

          )
    gr.on(triggers=[promt.submit, gen_button.click],
          fn=partial(inference, model="google/gemma-4-26B-A4B-it", model_name="gemma-4-26B-A4B-it"),
          inputs=[promt, token],
          outputs=[mistral_output],
          show_progress="hidden"

          )
    gr.on(triggers=[promt.submit, gen_button.click],
          fn=partial(inference, model="Qwen/Qwen2.5-72B-Instruct", model_name="Qwen2.5-72B-Instruct"),
          inputs=[promt, token],
          outputs=[qwen_output],
          show_progress="hidden"

          )

if __name__ == "__main__":
    demo.launch()
