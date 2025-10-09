# ollama_agent_tools.py
# pip install -U langchain langchain-ollama gradio

import os
import ast
import operator as op
import gradio as gr
import re

# -------------------------
#  Ollama Client Import
# -------------------------
try:
    from langchain_ollama import ChatOllama  # recommended new package
    OllamaClient = ChatOllama
except ImportError:
    from langchain_community.llms import Ollama  # fallback
    OllamaClient = Ollama

from langchain.agents import initialize_agent, Tool

# -------------------------
#  Safe Math Evaluator
# -------------------------
_ALLOWED_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
    ast.USub: op.neg,
}

def _safe_eval(expr: str):
    """
    Safely evaluate arithmetic expressions, including natural-language math.
    Supports +, -, *, /, %, **, unary minus.
    """
    expr = expr.strip().lower()

    # Remove quotes if present
    if (expr.startswith("'") and expr.endswith("'")) or (expr.startswith('"') and expr.endswith('"')):
        expr = expr[1:-1].strip()

    # -------------------
    # Normalize natural language
    # -------------------
    # Simple words
    expr = expr.replace("plus", "+").replace("minus", "-").replace("times", "*")
    expr = expr.replace("x", "*").replace("×", "*").replace("divided by", "/")
    expr = expr.replace("modulo", "%").replace("mod", "%").replace("power of", "**")
    expr = expr.replace("to the power of", "**")

    # Patterns like "add 5 and 12" → "5 + 12"
    add_match = re.match(r"add (\d+(\.\d+)?) and (\d+(\.\d+)?)", expr)
    if add_match:
        expr = f"{add_match[1]} + {add_match[3]}"

    sub_match = re.match(r"subtract (\d+(\.\d+)?) from (\d+(\.\d+)?)", expr)
    if sub_match:
        expr = f"{sub_match[3]} - {sub_match[1]}"

    mul_match = re.match(r"multiply (\d+(\.\d+)?) by (\d+(\.\d+)?)", expr)
    if mul_match:
        expr = f"{mul_match[1]} * {mul_match[3]}"

    div_match = re.match(r"divide (\d+(\.\d+)?) by (\d+(\.\d+)?)", expr)
    if div_match:
        expr = f"{div_match[1]} / {div_match[3]}"

    # Parse the final expression
    node = ast.parse(expr, mode="eval").body

    def _eval(n):
        if isinstance(n, ast.Constant):
            if isinstance(n.value, (int, float)):
                return n.value
            raise ValueError("Only numeric constants allowed.")
        if isinstance(n, ast.Num):
            return n.n
        if isinstance(n, ast.BinOp):
            op_type = type(n.op)
            if op_type not in _ALLOWED_OPS:
                raise ValueError(f"Operator {op_type} not allowed.")
            return _ALLOWED_OPS[op_type](_eval(n.left), _eval(n.right))
        if isinstance(n, ast.UnaryOp):
            op_type = type(n.op)
            if op_type not in _ALLOWED_OPS:
                raise ValueError(f"Unary operator {op_type} not allowed.")
            return _ALLOWED_OPS[op_type](_eval(n.operand))
        raise ValueError("Unsupported expression component: " + str(type(n)))

    return _eval(node)


# This **must be defined before** you reference it in tools
def calculator_tool(input_str: str) -> str:
    try:
        result = _safe_eval(input_str)
        return str(result)
    except Exception as e:
        return f"Calculator error: {e}"

# -------------------------
#  Local File Reader Tool
# -------------------------
def read_file_tool(path: str) -> str:
    path = os.path.expanduser(path.strip())
    if not os.path.isfile(path):
        return f"File not found: {path}"
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(20000)  # read first 20k chars
    except Exception as e:
        return f"Error reading file: {e}"

# -------------------------
#  Define Tools
# -------------------------
tools = [
    Tool(
        name="Calculator",
        func=calculator_tool,
        description="Useful for math. Input should be a single expression like '7 + 13' or '19*(2+3)'.",
    ),
    Tool(
        name="ReadFile",
        func=read_file_tool,
        description="Reads a local text file and returns its contents. Input should be a file path.",
    ),
]

# -------------------------
#  LLM Client (Ollama)
# -------------------------
llm = OllamaClient(model="llama3")  # change model if needed

# -------------------------
#  Initialize Agent
# -------------------------
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=6,
)

# -------------------------
#  Gradio Chat Interface
# -------------------------
def agent_chat(user_input: str, chat_history):
    try:
        result = agent.invoke(user_input)
        if isinstance(result, dict) and "output" in result:
            return result["output"]
        return str(result)
    except Exception as e:
        return f"Agent error: {e}"

if __name__ == "__main__":
    # Console test
    print("Quick test: What is 7 plus 13?")
    print(agent.invoke("What is 7 plus 13?"))

    # Launch Gradio UI
    gr.ChatInterface(fn=agent_chat, type="messages").launch(share=False)
