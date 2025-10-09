# pip install langgraph langchain langchain-community wikipedia duckduckgo-search
from langchain_ollama import ChatOllama
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# 1. Define LLM (replace with another if needed)
llm = ChatOllama(model="llama3")

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

# 3. Memory (LangGraph checkpoint)
memory = MemorySaver()

# 4. Create the agent with memory support
agent = create_react_agent(
    model=llm,
    tools=tools,
)

# 5. Simple REPL loop
if __name__ == "__main__":
    print("LangGraph Agent with Memory + Tools. Type 'exit' to quit.")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break

        # Run agent with checkpointing
        result = agent.invoke(
            {"messages": [("user", query)]},
            config={"configurable": {"thread_id": "chat1"}}  # thread_id links memory
        )

        print("Agent:", result["messages"][-1].content)
