import gradio as gr
from langchain_community.llms import Ollama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Initialize Ollama with llama3
llm = Ollama(model="llama3")

# Add memory so the AI remembers past turns
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory)

# Chat function
def chat(message, history):
    response = conversation.predict(input=message)
    return str(response)

if __name__ == "__main__":
    gr.ChatInterface(fn=chat).launch(share=True)
