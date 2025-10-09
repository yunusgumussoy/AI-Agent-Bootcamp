# ollama_agent_tools.py
# pip install wikipedia

import os, ast, operator as op, re
import gradio as gr

# -----------------------------------
# 1. Calculator Tool (Safe Math Eval)
# -----------------------------------
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
    expr = expr.strip().lower()
    expr = expr.replace("plus", "+").replace("minus", "-").replace("times", "*")
    expr = expr.replace("x", "*").replace("×", "*").replace("divided by", "/")
    expr = expr.replace("modulo", "%").replace("mod", "%").replace("to the power of", "**")

    add_match = re.match(r"add (\d+(\.\d+)?) and (\d+(\.\d+)?)", expr)
    if add_match:
        expr = f"{add_match[1]} + {add_match[3]}"

    mul_match = re.match(r"multiply (\d+(\.\d+)?) by (\d+(\.\d+)?)", expr)
    if mul_match:
        expr = f"{mul_match[1]} * {mul_match[3]}"

    node = ast.parse(expr, mode="eval").body

    def _eval(n):
        if isinstance(n, ast.Constant):
            return n.value if isinstance(n.value, (int, float)) else 0
        if isinstance(n, ast.Num):
            return n.n
        if isinstance(n, ast.BinOp):
            return _ALLOWED_OPS[type(n.op)](_eval(n.left), _eval(n.right))
        if isinstance(n, ast.UnaryOp):
            return _ALLOWED_OPS[type(n.op)](_eval(n.operand))
        raise ValueError("Unsupported expression")
    return _eval(node)

def calculator_tool(input_str: str) -> str:
    try:
        return str(_safe_eval(input_str))
    except Exception as e:
        return f"Calculator error: {e}"

# -----------------------------------
# 2. File Reader Tool
# -----------------------------------
def read_file_tool(path: str) -> str:
    path = os.path.expanduser(path.strip())
    if not os.path.isfile(path):
        return f"File not found: {path}"
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(20000)
    except Exception as e:
        return f"Error reading file: {e}"

# -----------------------------------
# 3. Wikipedia Tool
# -----------------------------------
from langchain_community.utilities import WikipediaAPIWrapper

wiki = WikipediaAPIWrapper(lang="en", top_k_results=1, doc_content_chars_max=500)


def wikipedia_tool(query: str) -> str:
    try:
        return wiki.run(query)
    except Exception as e:
        return f"Wikipedia error: {e}"

# -----------------------------------
# 4. Define Tools
# -----------------------------------
from langchain.agents import initialize_agent, Tool

tools = [
    Tool(
        name="Calculator",
        func=calculator_tool,
        description="Useful for math and arithmetic. Input like '7 + 13'."
    ),
    Tool(
        name="ReadFile",
        func=read_file_tool,
        description="Reads a local text file. Input a file path."
    ),
    Tool(
        name="Wikipedia",
        func=wikipedia_tool,
        description="Searches Wikipedia and returns a summary."
    )
]

# -----------------------------------
# 5. LLM Client (Ollama)
# -----------------------------------
try:
    from langchain_ollama import ChatOllama as _Ollama
    OllamaClient = _Ollama
except Exception:
    from langchain_community.llms import Ollama as OllamaClient

llm = OllamaClient(model="llama3", temperature=0)

# wrap LLM with a system message to guide it
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

system_prompt = """
You are an assistant that answers questions by calling tools when needed.
Always stop after giving the final answer.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

llm_chain = LLMChain(llm=llm, prompt=prompt)

# -----------------------------------
# 6. Initialize Agent
# -----------------------------------

agent = initialize_agent(
    tools,
    llm, # llm_chain
    agent="chat-zero-shot-react-description",  # <-- chat agent
    verbose=False, # True
    handle_parsing_errors=True,
    max_iterations=6,
)

# -----------------------------------
# 7. Gradio Chat Interface
# -----------------------------------


def agent_chat(user_input: str, chat_history):
    try:
        result = agent.invoke(user_input)
        content = result["output"] if isinstance(result, dict) and "output" in result else str(result)
    except Exception as e:
        content = f"Agent error: {e}"

    # Add the new user/assistant turn to the history
    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": content})
    return chat_history


if __name__ == "__main__":
    print("Quick test:")
    print(agent.invoke("What is 7 plus 13?"))
    print(agent.invoke("Who is Albert Einstein?"))
    gr.ChatInterface(fn=agent_chat, type="messages").launch(share=False)