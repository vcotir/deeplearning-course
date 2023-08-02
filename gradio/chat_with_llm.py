# import os
import io
import IPython.display
from PIL import Image
import base64 
import requests 
requests.adapters.DEFAULT_TIMEOUT = 60

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
hf_api_key = os.environ['HF_API_KEY']

# Helper function
import requests, json
from text_generation import Client

#FalcomLM-instruct endpoint on the text_generation library
client = Client(os.environ['HF_API_FALCOM_BASE'], headers={"Authorization": f"Basic {hf_api_key}"}, timeout=120)

'''
Here we'll be using an [Inference Endpoint](https://huggingface.co/inference-endpoints) for `falcon-40b-instruct` , one of best ranking open source LLM on the [🤗 Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard). 

To run it locally, one can use the [Transformers library](https://huggingface.co/docs/transformers/index) or the [text-generation-inference](https://github.com/huggingface/text-generation-inference) 
'''

prompt = "Has math been invented or discovered?"
client.generate(prompt, max_new_tokens=256).generated_text

#Back to Lesson 2, time flies!
import gradio as gr
def generate(input, slider):
    output = client.generate(input, max_new_tokens=slider).generated_text
    return output

demo = gr.Interface(fn=generate, inputs=[gr.Textbox(label="Prompt"), gr.Slider(label="Max new tokens", value=20,  maximum=1024, minimum=1)], outputs=[gr.Textbox(label="Completion")])
gr.close_all()
demo.launch(share=True, server_port=int(os.environ['PORT1']))

# gr.Chatbot can help streamline resending context
import random

def respond(message, chat_history):
        #No LLM here, just respond with a random pre-made message
        bot_message = random.choice(["Tell me more about it", 
                                     "Cool, but I'm not interested", 
                                     "Hmmmm, ok then"]) 
        chat_history.append((message, bot_message))
        return "", chat_history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(height=240) #just to fit the notebook
    msg = gr.Textbox(label="Prompt")
    btn = gr.Button("Submit")
    clear = gr.ClearButton(components=[msg, chatbot], value="Clear console")

    btn.click(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
    msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot]) #Press enter to submit
gr.close_all()
demo.launch(share=True, server_port=int(os.environ['PORT2']))