from dotenv import load_dotenv
import os
load_dotenv()

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, SystemMessage
from huggingface_hub import InferenceClient

print("Testing direct InferenceClient...")
client = InferenceClient("meta-llama/Meta-Llama-3-8B-Instruct", token=os.environ.get("HUGGINGFACEHUB_API_TOKEN"))

try:
    response = client.chat_completion(
        messages=[{"role": "user", "content": "ping"}]
    )
    print("Success InferenceClient:", response.choices[0].message.content)
except Exception as e:
    print(f"Failed InferenceClient: {type(e).__name__} - {e}")
