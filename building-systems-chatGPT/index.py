import os 
import openai 
import tiktoken
from dotenv import load_dotenv, find_dotenv 
_ = load_dotenv(find_dotenv()) # read .env 

openai.api_key = os.environ['OPENAI_API_KEY']

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message["content"]