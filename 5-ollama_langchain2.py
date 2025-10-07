from langchain_ollama import ChatOllama
from langchain.agents import initialize_agent, Tool

def adder_tool(input_str: str) -> str:
    try:
        numbers = [int(s) for s in input_str.replace("and", " ").replace("plus", " ").split() if s.isdigit()]
        if len(numbers) == 2:
            return str(sum(numbers))
        return "Please provide two numbers like: '7 13'"
    except Exception as e:
        return f"Error: {e}"

'''
tools = [
    Tool(
        name="Adder",
        func=adder_tool,
        description="Adds two numbers. Example inputs: '7 13', '7 and 13', '7 plus 13'."
    )
]
'''
tools = [
    Tool(
        name="Adder",
        func=adder_tool,
        description=(
            "Adds two numbers. "
            "Input must be two integers separated by a space, e.g., '7 13'. "
            "Always respond in the format: "
            "Action: Adder\nAction Input: 7 13"
        )
    )
]


llm = ChatOllama(model="llama3")

# agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True,
    handle_parsing_errors=True
)

response = agent.invoke("What is 7 plus 13?")
print(response)
