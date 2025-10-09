# pip install langgraph langchain langchain-community wikipedia
# pip install -U ddgs

from langchain_ollama import ChatOllama
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# 1. Define LLM (you can replace with OpenAI if you want)
llm = ChatOllama(model="llama3")  # change to "mistral" or any model you have

# 2. Define Tools
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
search = DuckDuckGoSearchRun()

tools = [
    Tool.from_function(
        func=lambda x: str(eval(x)),  # simple calculator
        name="Calculator",
        description="Useful for math operations"
    ),
    Tool.from_function(
        func=wiki.run,
        name="Wikipedia",
        description="Useful for getting factual knowledge"
    ),
    Tool.from_function(
        func=search.run,
        name="DuckDuckGo",
        description="Useful for web search queries"
    )
]

# 3. Add memory (LangGraph style: checkpointing)
memory = MemorySaver()

# 4. Create the agent
agent = create_react_agent(
    model=llm,
    tools=tools
)

# 5. Simple REPL loop
if __name__ == "__main__":
    print("LangGraph Agent with Memory + Tools. Type 'exit' to quit.")
    state = None
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break

        # Run agent (stateful)
        result = agent.invoke({"messages": [("user", query)]}, config={"configurable": {"thread_id": "chat1"}})
        print("Agent:", result["messages"][-1].content)
