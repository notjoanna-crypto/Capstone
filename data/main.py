from datapizza.clients.openai import OpenAIClient
import os
from dotenv import load_dotenv

# Run code with: docker compose exec app python data/main.py


load_dotenv()

client = OpenAIClient(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-5",
    system_prompt="You are a helpful assistant that can answer questions and help with tasks.",
)

response = client.invoke("Explain the concept of quantum computing in one sentence.")
print(response.text)

