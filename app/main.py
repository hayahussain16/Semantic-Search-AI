import asyncio
import streamlit as st

from app.analyzer import analyze_document, ask_document
from app.document_parser import extract_text
from app.llm_client import LLMError, ProductionLLMClient


st.set_page_config(
    page_title="Document Intelligence Assistant",
    layout="wide"
)

st.title("📄 AI Document Intelligence Assistant")

client = ProductionLLMClient()

uploaded = st.file_uploader(
    "Upload a PDF or TXT file",
    type=["pdf", "txt"]
)

if uploaded and st.button("Analyze document"):
    text = extract_text(uploaded.name, uploaded.read())

    if not text.strip():
        st.error("No text could be extracted from this file.")
    else:
        with st.spinner("Analyzing..."):
            try:
                analysis, stats, working_text = asyncio.run(
                    analyze_document(client, text)
                )

                st.session_state.analysis = analysis
                st.session_state.stats = stats
                st.session_state.working_text = working_text

            except LLMError as e:
                st.error(
                    f"The model could not produce a valid result: {e}"
                )

if "analysis" in st.session_state:
    a = st.session_state.analysis

    st.subheader("Summary")
    st.write(a.summary)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Key points")
        [st.write(f"- {x}") for x in a.key_points]

        st.subheader("Risks")
        [st.write(f"- {x}") for x in a.risks]

    with col2:
        st.subheader("Action items")
        [st.write(f"- {x}") for x in a.action_items]

        st.subheader("Entities")
        st.write(", ".join(a.entities) or "None found")

    st.subheader("Structured JSON")
    st.json(a.model_dump())

    st.subheader("Run stats")
    st.json(st.session_state.stats)

    st.subheader("Ask a question about the document")

    q = st.text_input("Your question")

    if q:
        with st.spinner("Thinking..."):
            answer = asyncio.run(
                ask_document(
                    client,
                    st.session_state.working_text,
                    q
                )
            )

        st.write(answer)