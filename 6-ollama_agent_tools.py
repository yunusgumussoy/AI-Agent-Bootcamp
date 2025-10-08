# pip install langchain-experimental

import re
import gradio as gr
from langchain.agents import initialize_agent, Tool
from langchain_community.llms import Ollama
from langchain_experimental.utilities.python import PythonREPL  # ✅ Updated import


# -------------------
# Math Expression Parser
# -------------------
def parse_math_expression(query: str) -> str:
    query = query.lower()
    query = query.replace("plus", "+")
    query = query.replace("minus", "-")
    query = query.replace("times", "*")
    query = query.replace("multiplied by", "*")
    query = query.replace("divided by", "/")
    expr = re.sub(r"[^0-9+\-*/(). ]", "", query)
    return expr.strip()


# -------------------
# Safe Calculator Wrapper
# -------------------
python_repl = PythonREPL()

def safe_calculator(query: str) -> str:
    expr = parse_math_expression(query)
    if expr:
        try:
            return python_repl.run(expr)
        except Exception as e:
            return f"Calculator error: {e}"
    else:
        return "I couldn’t parse a math expression."


# -------------------
# LLM Setup
# -------------------
llm = Ollama(model="llama3")

tools = [
    Tool(
        name="Calculator",
        func=safe_calculator,
        description="Useful for solving math problems. Example: '7 plus 13'."
    )
]

agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True,
    handle_parsing_errors=True
)


# -------------------
# Chat Function
# -------------------
def chat(message, history):
    try:
        return agent.run(message)
    except Exception as e:
        return f"Error: {e}"


# -------------------
# Launch Gradio Chat
# -------------------
gr.ChatInterface(fn=chat, type="messages").launch(share=True)
