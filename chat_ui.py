import gradio as gr
from langchain_community.llms import Ollama

llm = Ollama(model="llama3")

def chat(message, history):
    # Combine history into a conversational prompt
    context = "\n".join([f"User: {u}\nAI: {a}" for u, a in history])
    prompt = f"{context}\nUser: {message}\nAI:"
    response = llm.invoke(prompt)
    return str(response)

if __name__ == "__main__":
    gr.ChatInterface(fn=chat).launch(share=True)

