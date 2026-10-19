import ast
import json
import operator as op
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY")
)

MODEL = os.getenv("LLM_MODEL")


# ---------- 1. The tools (plain Python functions) ----------

OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
    ast.USub: op.neg
}


def calculator(expression: str):
    """Safe maths. NEVER use eval() on text that came from a model."""

    def _eval(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value

        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return OPS[type(node.op)](
                _eval(node.left),
                _eval(node.right)
            )

        if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
            return OPS[type(node.op)](
                _eval(node.operand)
            )

        raise ValueError("Unsupported expression")

    return _eval(ast.parse(expression, mode="eval").body)


WEATHER = {
    "hyderabad": "31°C, partly cloudy",
    "mumbai": "29°C, humid",
    "delhi": "35°C, sunny"
}


def get_weather(city: str):
    return WEATHER.get(
        city.lower(),
        "No data for that city"
    )


KNOWLEDGE = {
    "token": "A token is a chunk of text (about 3/4 of a word) that an LLM reads.",
    "embedding": "An embedding is a vector that represents the meaning of text.",
    "rag": "RAG retrieves relevant documents and gives them to the LLM before it answers."
}


def knowledge_lookup(topic: str):
    return KNOWLEDGE.get(
        topic.lower(),
        "No entry found for that topic"
    )


TOOL_REGISTRY = {
    "calculator": calculator,
    "get_weather": get_weather,
    "knowledge_lookup": knowledge_lookup,
}


# ---------- 2. The schemas the model sees ----------

def tool(name, description, param, param_desc):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    param: {
                        "type": "string",
                        "description": param_desc
                    }
                },
                "required": [param]
            }
        }
    }


TOOLS = [
    tool(
        "calculator",
        "Evaluate arithmetic such as 25*438. Use for ANY maths.",
        "expression",
        "The maths expression"
    ),
    tool(
        "get_weather",
        "Get the current weather for a city.",
        "city",
        "City name, for example Hyderabad"
    ),
    tool(
        "knowledge_lookup",
        "Look up a short definition of an AI term.",
        "topic",
        "The term, for example token"
    ),
]


# ---------- 3. The loop ----------

def run_assistant(user_message: str, max_steps: int = 5) -> str:

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Use tools when they help. "
                "If no tool is needed, answer directly."
            )
        },
        {
            "role": "user",
            "content": user_message
        },
    ]

    for _ in range(max_steps):  # guard against infinite loops

        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )

        msg = resp.choices[0].message

        if not msg.tool_calls:  # model gave a final answer
            return msg.content

        messages.append(msg)  # keep the model's tool request

        for call in msg.tool_calls:

            name = call.function.name

            try:
                args = json.loads(call.function.arguments)
                result = TOOL_REGISTRY[name](**args)

            except Exception as e:
                # tool errors go back to the model
                result = f"Tool error: {e}"

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": str(result)
            })

    return "Stopped: too many tool steps."


if __name__ == "__main__":

    for q in [
        "What is 25 x 438?",
        "What's the weather in Hyderabad?",
        "What is an embedding? Also compute 12*12.",
        "Tell me a joke."
    ]:
        print(f"\nQ: {q}\nA: {run_assistant(q)}")