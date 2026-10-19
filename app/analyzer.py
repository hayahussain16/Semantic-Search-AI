import asyncio
import time

from app.document_parser import chunk_text, count_tokens
from app.llm_client import ProductionLLMClient
from app.prompts import (
    build_analysis_messages,
    build_chunk_summary_messages,
    build_qa_messages,
)
from app.schemas import DocumentAnalysis

MAX_DIRECT_TOKENS = 6000


async def analyze_document(client: ProductionLLMClient, text: str):
    start = time.perf_counter()

    tokens_in = count_tokens(text)

    if tokens_in > MAX_DIRECT_TOKENS:
        chunks = chunk_text(text)

        summaries = await asyncio.gather(
            *[
                client.chat(
                    build_chunk_summary_messages(c),
                    temperature=0
                )
                for c in chunks
            ]
        )

        working_text = "\n\n".join(summaries)
    else:
        working_text = text

    analysis = await client.generate_structured(
        build_analysis_messages(working_text),
        DocumentAnalysis
    )

    stats = {
        "input_tokens_estimate": tokens_in,
        "chunks": 1 if tokens_in <= MAX_DIRECT_TOKENS else len(chunks),
        "latency_s": round(time.perf_counter() - start, 2),
        "validation_passed": True,
    }

    return analysis, stats, working_text


async def ask_document(
    client: ProductionLLMClient,
    working_text: str,
    question: str
) -> str:
    return await client.chat(
        build_qa_messages(working_text, question),
        temperature=0
    )