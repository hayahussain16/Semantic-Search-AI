import pandas as pd
from app.evaluator import run_eval

cases = [
    {
        "question": "What is Python?",
        "expected_keywords": ["programming", "language"],
    },
    {
        "question": "What is an API?",
        "expected_keywords": ["interface", "software"],
    },
    {
        "question": "What is a token in an LLM?",
        "expected_keywords": ["text", "LLM"],
    },
    {
        "question": "What is an embedding?",
        "expected_keywords": ["vector", "meaning"],
    },
    {
        "question": "What is RAG?",
        "expected_keywords": ["retrieves", "documents"],
    },
    {
        "question": "What is machine learning?",
        "expected_keywords": ["data", "learning"],
    },
    {
        "question": "What is a neural network?",
        "expected_keywords": ["neurons", "network"],
    },
    {
        "question": "What is semantic search?",
        "expected_keywords": ["meaning", "similarity"],
    },
    {
        "question": "What is function calling?",
        "expected_keywords": ["tool", "function"],
    },
    {
        "question": "What is JSON?",
        "expected_keywords": ["data", "format"],
    },
    {
        "question": "What is Pydantic?",
        "expected_keywords": ["validation", "data"],
    },
    {
        "question": "What is temperature in an LLM?",
        "expected_keywords": ["randomness", "output"],
    },
    {
        "question": "What is top-p?",
        "expected_keywords": ["probability", "tokens"],
    },
    {
        "question": "What is hallucination in an LLM?",
        "expected_keywords": ["false", "information"],
    },
    {
        "question": "Why is evaluation important for LLM applications?",
        "expected_keywords": ["correctness", "reliability"],
    },
    {
        "question": "What is cosine similarity?",
        "expected_keywords": ["vectors", "similarity"],
    },
    {
        "question": "Why should we avoid eval() on model output?",
        "expected_keywords": ["unsafe", "code"],
    },
    {
        "question": "What is structured output?",
        "expected_keywords": ["JSON", "schema"],
    },
    {
        "question": "What is prompt engineering?",
        "expected_keywords": ["prompt", "instructions"],
    },
    {
        "question": "Why should LLM outputs be validated?",
        "expected_keywords": ["validation", "errors"],
    },
    {
        "question": "Who won the 2087 World Cup?",
        "expected_keywords": ["don't know", "no information"],
    },
]

frames = [run_eval(m, cases) for m in ["llama3.2", "phi3:mini"]]

df = pd.concat(frames)

report = df.groupby("model").agg(
    accuracy=("correctness", "mean"),
    avg_latency_s=("latency_s", "mean"),
    total_tokens=("tokens", "sum"),
    failure_rate=("failed", "mean"),
).round(3)

print(report)

df.to_csv("eval_results.csv", index=False)