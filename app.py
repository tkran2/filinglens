"""FilingLens: financial-document questions with inspectable evidence."""

import json

import streamlit as st
from openai import APIConnectionError, APIStatusError

from filinglens.answer import ROOT, generate_answer
from filinglens.limits import DemoLimiter
from filinglens.search import KeywordSearch, load_chunks

REPORT_URL = (
    "https://www.annualreports.com/HostedData/"
    "AnnualReportArchive/a/NASDAQ_AAPL_2024.pdf"
)

st.set_page_config(
    page_title="FilingLens | Financial document research",
    page_icon="📄",
    layout="wide",
)


@st.cache_resource
def get_retriever(index_signature):
    chunks = load_chunks(ROOT / "assets")
    if not chunks:
        raise ValueError("Deployment passages are missing.")
    return KeywordSearch(chunks)


@st.cache_resource
def get_usage_limiter():
    return DemoLimiter()


def report_link(evidence):
    if evidence["source"] == "apple-2024.pdf":
        return f"{REPORT_URL}#page={evidence['page']}"
    return None


st.caption("FILINGLENS / FINANCIAL DOCUMENT RESEARCH")
st.title("Answers you can trace to the filing.")
st.write(
    "Ask a question about Apple's 2024 annual report. "
    "Read the answer, then inspect the passages behind it."
)

with st.sidebar:
    st.subheader("Document collection")
    st.write("**Apple Inc.**")
    st.caption("2024 Form 10-K · 121 PDF pages")
    st.link_button("Open annual report", REPORT_URL)
    st.divider()
    st.write("**Try asking**")
    st.caption("Why are sales higher in the first fiscal quarter?")
    st.caption("How could tariffs affect Apple's business?")
    st.caption("What risks arise from manufacturing defects?")
    st.divider()
    st.caption(
        "Answers use retrieved passages. Citations make checking easier "
        "but do not guarantee correctness."
    )

with st.form("question_form"):
    question = st.text_input(
        "Your question",
        value="Why does Apple tend to have higher sales in its first fiscal quarter?",
        max_chars=1000,
    )
    submitted = st.form_submit_button("Find an answer", type="primary")

if submitted:
    st.session_state.pop("result", None)
    question = question.strip()

    if not question:
        st.warning("Enter a question first.")
    else:
        try:
            index_files = sorted((ROOT / "assets").glob("*.json"))
            if not index_files:
                raise ValueError("No documents indexed. Run PDF ingestion first.")

            signature = tuple(
                (p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in index_files
            )

            with st.spinner("Finding relevant passages..."):
                retriever = get_retriever(signature)
                passages = retriever.search(question, top_k=5)

            if not passages:
                st.info("No matching passages were found.")
            else:
                with st.spinner("Writing an answer from the evidence..."):
                    get_usage_limiter().acquire()
                    st.session_state["result"] = generate_answer(question, passages)

        except APIStatusError as error:
            st.error(
                f"The model service returned HTTP {error.status_code}. "
                "Check API billing, key permissions, or model availability."
            )
        except APIConnectionError:
            st.error("Could not reach the model service. Check your connection.")
        except ValueError as error:
            st.error(str(error))

result = st.session_state.get("result")

if result:
    st.divider()
    st.subheader(result["question"])
    answer_column, evidence_column = st.columns([3, 2], gap="large")
    evidence_by_id = {item["citation"]: item for item in result["evidence"]}

    with answer_column:
        st.subheader("Answer")
        if not result["answer"]["supported"]:
            st.info("The retrieved evidence does not answer this question.")
        else:
            for claim in result["answer"]["claims"]:
                st.write(claim["text"])
                links = []
                for number in sorted(set(claim["citations"])):
                    evidence = evidence_by_id[number]
                    url = report_link(evidence)
                    label = f"[{number}] PDF page {evidence['page']}"
                    links.append(f"[{label}]({url})" if url else label)
                st.markdown(" · ".join(links))

        st.caption(f"Answer generation: {result['generation_seconds']:.2f} seconds")
        st.download_button(
            "Download answer and evidence",
            data=json.dumps(result, indent=2),
            file_name="filinglens-answer.json",
            mime="application/json",
        )

    with evidence_column:
        st.subheader("Retrieved evidence")
        used = {
            number
            for claim in result["answer"]["claims"]
            for number in claim["citations"]
        }
        for evidence in result["evidence"]:
            number = evidence["citation"]
            status = "Cited" if number in used else "Retrieved"
            with st.expander(
                f"[{number}] {status} · PDF page {evidence['page']}",
                expanded=number in used,
            ):
                st.write(evidence["text"])
                url = report_link(evidence)
                if url:
                    st.link_button(f"Open PDF page {evidence['page']}", url)
else:
    st.info("Ask a question to see an answer alongside its source passages.")
