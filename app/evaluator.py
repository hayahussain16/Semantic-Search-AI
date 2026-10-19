import json
import os
import time

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY")
)


def keyword_score(answer: str, keywords: list[str]) -> float:
    """Fraction of expected keywords found (case-insensitive)."""

    answer = answer.lower()

    hits = sum(
        1 for k in keywords
        if k.lower() in answer
    )

    return hits / len(keywords) if keywords else 0.0


def is_valid_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except Exception:
        return False


def run_eval(
    model: str,
    cases: list[dict],
    system: str = "Answer briefly and accurately."
) -> pd.DataFrame:

    rows = []

    for case in cases:

        start = time.perf_counter()

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": case["question"]}
                ],
                temperature=0,
            )

            answer = resp.choices[0].message.content or ""

            tokens = getattr(
                resp.usage,
                "total_tokens",
                0
            )

            failed = answer.strip() == ""

        except Exception as e:

            answer = f"ERROR: {e}"
            tokens = 0
            failed = True

        rows.append({
            "model": model,
            "question": case["question"],
            "answer": answer[:120],
            "correctness": (
                0.0
                if failed
                else keyword_score(
                    answer,
                    case["expected_keywords"]
                )
            ),
            "latency_s": round(
                time.perf_counter() - start,
                2
            ),
            "tokens": tokens,
            "failed": failed,
        })

    return pd.DataFrame(rows)

from pydantic import BaseModel, Field


class JudgeResult(BaseModel):
    score: int = Field(ge=1, le=5)
    reason: str


JUDGE_SYSTEM = """You are a strict grader.
You get a CONTEXT, a QUESTION and an ANSWER.
Rate GROUNDEDNESS from 1 to 5:
5 = every claim in the ANSWER is supported by the CONTEXT
3 = partly supported
1 = mostly unsupported or invented
Return ONLY JSON: {"score": <1-5>, "reason": "<one sentence>"}"""


def judge_groundedness(
    judge_model: str,
    context: str,
    question: str,
    answer: str
) -> JudgeResult:

    resp = client.chat.completions.create(
        model=judge_model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": JUDGE_SYSTEM
            },
            {
                "role": "user",
                "content": (
                    f"CONTEXT:\n{context}\n\n"
                    f"QUESTION:\n{question}\n\n"
                    f"ANSWER:\n{answer}"
                )
            }
        ],
    )

    raw = resp.choices[0].message.content

    return JudgeResult.model_validate_json(
        raw[raw.find("{"): raw.rfind("}") + 1]
    )