# pip install langchain langchain-community langchain-ollama wikipedia duckduckgo-search gradio PyPDF2 docx2txt
# pip install PyPDF2
# pip install docx2txt

import gradio as gr
import os
import PyPDF2
import docx2txt
from langchain_ollama import ChatOllama
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool


# === 1. Define LLM (Ollama model, fully local)
llm = ChatOllama(model="llama3", temperature=0.3)

# === 2. Safe Calculator Tool
def safe_calculator(expression: str) -> str:
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"Calculator error: {e}"

# === 3. File Reader Tool
DOCUMENT_CONTEXT = ""

def read_file(file_path: str) -> str:
    global DOCUMENT_CONTEXT
    text = ""

    if not os.path.exists(file_path):
        return "File not found."

    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext == ".pdf":
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() or ""
        elif ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        elif ext == ".docx":
            text = docx2txt.process(file_path)
        else:
            return "Unsupported file format. Please upload PDF, TXT, or DOCX."

        DOCUMENT_CONTEXT = text[:10000]  # limit context for safety
        return "File uploaded and processed successfully."

    except Exception as e:
        return f"Error reading file: {e}"


def ask_about_file(question: str) -> str:
    global DOCUMENT_CONTEXT
    if not DOCUMENT_CONTEXT:
        return "Please upload a file first."
    prompt = f"Based on the following document, answer the question.\n\nDocument:\n{DOCUMENT_CONTEXT}\n\nQuestion: {question}\nAnswer:"
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Error processing question: {e}"


# === 4. Wikipedia & Search Tools
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
search = DuckDuckGoSearchRun()

tools = [
    Tool.from_function(func=safe_calculator, name="Calculator", description="Performs basic math"),
    Tool.from_function(func=wiki.run, name="Wikipedia", description="Looks up facts"),
    Tool.from_function(func=search.run, name="DuckDuckGo", description="Searches the web"),
    Tool.from_function(func=read_file, name="ReadFile", description="Reads and stores document text (PDF, TXT, DOCX)"),
    Tool.from_function(func=ask_about_file, name="AskFile", description="Answer questions based on uploaded document")
]

# === 5. Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# === 6. Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# === 7. Gradio UI (Final Fixed Version)
def chat_with_agent(user_message, history, file=None):
    global DOCUMENT_CONTEXT

    # Step 1: Handle file upload (only when it's a *new* file)
    if file is not None and file.name:
        result = read_file(file.name)
        DOCUMENT_CONTEXT = DOCUMENT_CONTEXT.strip()
        return f"{result}\n✅ File processed successfully! Now ask questions like:\n- 'Summarize the document'\n- 'What is the main topic?'"

    # Step 2: If there is document context, answer from it
    if DOCUMENT_CONTEXT:
        reply = ask_about_file(user_message)
    else:
        try:
            reply = agent.run(user_message)
        except Exception as e:
            reply = f"Agent error: {e}"

    return reply


# === 8. Gradio UI Layout
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Local AI Agent with Memory + File Reading (Ollama)")
    chatbot = gr.Chatbot()
    with gr.Row():
        msg = gr.Textbox(label="Your message")
        file_upload = gr.File(label="Upload a file (PDF, TXT, DOCX)", file_types=[".pdf", ".txt", ".docx"])
    clear = gr.Button("Clear")

    # Gradio handler
    def respond(message, chat_history, file):
        reply = chat_with_agent(message, chat_history, file)
        chat_history.append((message, reply))
        return "", chat_history, None  # Clear file after processing

    msg.submit(respond, [msg, chatbot, file_upload], [msg, chatbot, file_upload])
    clear.click(lambda: None, None, chatbot, queue=False)

    gr.Markdown("💬 Tip: Upload a document and then ask questions about it.")

demo.launch(share=False)
