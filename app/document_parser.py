import io
import tiktoken
from pypdf import PdfReader

ENC = tiktoken.get_encoding("cl100k_base")


def extract_text(filename: str, data: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((page.extract_text() or "") for page in reader.pages)

    return data.decode("utf-8", errors="ignore")


def count_tokens(text: str) -> int:
    return len(ENC.encode(text))


def chunk_text(text: str, max_tokens: int = 3000) -> list[str]:
    """Split by paragraphs, keeping each chunk under max_tokens."""
    chunks, current, current_tokens = [], [], 0

    for para in text.split("\n"):
        t = count_tokens(para)

        if current_tokens + t > max_tokens and current:
            chunks.append("\n".join(current))
            current, current_tokens = [], 0

        current.append(para)
        current_tokens += t

    if current:
        chunks.append("\n".join(current))

    return chunks