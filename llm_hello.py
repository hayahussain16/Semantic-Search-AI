import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY")
)

model = os.getenv("LLM_MODEL")

response = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "system",
            "content": "You are a friendly teacher."
        },
        {
            "role": "user",
            "content": "Explain what a token is in an LLM."
        }
    ],
    temperature=1.2,
    max_tokens=200
)
print(response.choices[0].message.content)
print("\nUsage:", response.usage)