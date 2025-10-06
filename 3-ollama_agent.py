from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Connect to local Ollama (default model = llama3)
llm = Ollama(model="llama3")

# Define a simple agent prompt
prompt = PromptTemplate(
    input_variables=["task"],
    template="You are a helpful AI() agent. Please complete the following task:\n{task}"
)

# Create a chain
chain = LLMChain(llm=llm, prompt=prompt)

# Example: Run a task
task = "Suggest 3 creative startup ideas for AI-powered apps."
result = chain.run(task)

print("Agent Response:\n", result)
