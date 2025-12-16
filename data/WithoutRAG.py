from datapizza.clients.openai import OpenAIClient
import os
from dotenv import load_dotenv

# Run code with: docker compose exec app python data/main.py

query = "Vilket är det vanligaste färdmedlet bland de med en sammanlagd hushållsinkomst under 10 000 kronor per månad? "

load_dotenv()

client = OpenAIClient(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini",
    system_prompt="You are a helpful assistant that can answer questions and help with tasks.",
)

response = client.invoke(query)
print(response.text)

