# from langchain_ollama import ChatOllama
from langchain_community.chat_models import ChatOllama
from langchain.agents import initialize_agent, Tool
from langchain import hub


# 1. Connect to local Ollama
# llm = ChatOllama(model="llama2")
llm = ChatOllama(
    model="llama3",  # Or "llama2", "mistral", etc. (must be pulled in Ollama first)
    base_url="http://localhost:11434"
)

# 2. Example tool
def add_numbers(x: str, y: str) -> str:
    return str(int(x) + int(y))

tools = [
    Tool(
        name="Adder",
        func=lambda q: add_numbers(*q.split()),
        description="Add two integers. Input should be 'x y'."
    )
]

# 3. Initialize agent with tools
agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",  # reasoning agent
    verbose=True
)

# 4. Run agent
print("---- Asking agent ----")
response = agent.run("What is 7 plus 13?")
print(response)
