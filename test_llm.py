from dotenv import load_dotenv
load_dotenv()
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, SystemMessage

try:
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        task="text-generation",
        max_new_tokens=512,
        temperature=0.1,
    )
    chat_model = ChatHuggingFace(llm=llm)

    messages = [
        SystemMessage(content="You are a helpful AI cybersecurity assistant. Context: Test"),
        HumanMessage(content="Test question")
    ]
    response = chat_model.invoke(messages)
    print("Success:", response.content)
except Exception as e:
    import traceback
    print("Error:", traceback.format_exc())
