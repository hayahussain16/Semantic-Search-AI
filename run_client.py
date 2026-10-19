import asyncio
import logging

from app.llm_client import ProductionLLMClient
from app.prompts import build_customer_messages
from app.schemas import CustomerAnalysis


logging.basicConfig(level=logging.INFO)


async def main():
    client = ProductionLLMClient()

    messages = build_customer_messages(
        "Asha",
        "Third time my order arrived late. I am thinking of cancelling."
    )

    result = await client.generate_structured(
        messages,
        CustomerAnalysis
    )

    print(result)
    print(result.churn_risk)


asyncio.run(main())