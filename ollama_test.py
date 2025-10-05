import requests
import json

# Ollama API endpoint
url = "http://localhost:11434/api/generate"

# Define the model and the prompt
payload = {
    "model": "llama3",
    "prompt": "Write a one-sentence definition of an AI agent."
}

# Send POST request
response = requests.post(url, json=payload, stream=True)

# Process streaming response
for line in response.iter_lines():
    if line:
        data = json.loads(line.decode("utf-8"))
        if "response" in data:
            print(data["response"], end="", flush=True)

print("\n---\nDone.")
