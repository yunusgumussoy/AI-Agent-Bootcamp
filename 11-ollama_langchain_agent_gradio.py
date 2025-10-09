# pip install langchain langchain-community langchain-ollama wikipedia duckduckgo-search gradio

import gradio as gr
from langchain_ollama import ChatOllama
from langchain.agents import initialize_agent, AgentType
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool


# 1. Define Local LLM (Free via Ollama)
llm = ChatOllama(
    model="llama3",   # you can switch to "mistral" or "phi3" if installed
    temperature=0.2
)

# 2. Define Tools
def safe_calculator(expression: str) -> str:
    """Safely evaluate basic math expressions."""
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"Calculator error: {e}"

wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
search = DuckDuckGoSearchRun()

tools = [
    Tool.from_function(
        func=safe_calculator,
        name="Calculator",
        description="Useful for math operations (e.g., '7+13' or '23*5')"
    ),
    Tool.from_function(
        func=wiki.run,
        name="Wikipedia",
        description="Useful for factual knowledge"
    ),
    Tool.from_function(
        func=search.run,
        name="DuckDuckGo",
        description="Useful for general web search queries"
    )
]

# 3. Create the Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # ReAct reasoning agent
    verbose=True
)

# 4. Define Gradio Chat Interface
def chat_with_agent(user_message, history):
    try:
        response = agent.run(user_message)
    except Exception as e:
        response = f"Agent error: {e}"
    return response


# 5. Launch Gradio App
if __name__ == "__main__":
    print("🚀 Launching Gradio Chat at http://127.0.0.1:7860 ...")
    gr.ChatInterface(fn=chat_with_agent, title="🧠 Local AI Agent (Ollama + LangChain)").launch(share=False)
