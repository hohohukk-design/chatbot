"""DATA 폴더의 PDF를 이용한 간단한 RAG 챗봇입니다."""

from __future__ import annotations

import os
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


# 프로젝트 최상위 폴더와 문서 폴더를 기준으로 경로를 계산합니다.
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "DATA"

# .env에 저장한 OPENAI_API_KEY를 환경 변수로 불러옵니다.
load_dotenv(PROJECT_ROOT / ".env")


def read_pdf_documents() -> list[Document]:
    """DATA 폴더의 모든 PDF를 페이지 단위 Document로 읽습니다."""

    documents: list[Document] = []
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))

    for pdf_path in pdf_files:
        reader = PdfReader(str(pdf_path))
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path.name,
                        "page": page_number,
                    },
                )
            )

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """긴 페이지를 검색하기 좋은 크기의 조각으로 나눕니다."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1_000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", "。", "! ", "? ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks, start=1):
        # 검증 보고서에서 어떤 청크를 확인했는지 추적할 수 있도록 식별자를 붙입니다.
        chunk.metadata["chunk_id"] = f"{chunk.metadata['source']}:{chunk.metadata['page']}:{index}"
    return chunks


@st.cache_resource(show_spinner="문서를 임베딩하고 검색 인덱스를 만드는 중입니다...")
def build_vector_store(api_key: str) -> tuple[InMemoryVectorStore, int, int]:
    """문서를 임베딩해 InMemoryVectorStore에 한 번만 저장합니다."""

    page_documents = read_pdf_documents()
    chunks = split_documents(page_documents)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
    )
    vector_store = InMemoryVectorStore(embedding=embeddings)
    vector_store.add_documents(chunks)
    return vector_store, len(page_documents), len(chunks)


def make_evidence_sentence(text: str) -> str:
    """검색된 조각에서 화면에 보여줄 근거 문장을 뽑습니다."""

    cleaned = re.sub(r"\s+", " ", text).strip()
    sentences = [part.strip() for part in re.split(r"(?<=[.!?。！？])\s+", cleaned)]
    sentences = [sentence for sentence in sentences if sentence]
    if not sentences:
        return cleaned[:300]

    # 너무 짧은 머리말보다 실제 내용이 있는 문장을 우선합니다.
    return max(sentences[:4], key=len)[:500]


def response_text(response: object) -> str:
    """LangChain 응답을 화면에 표시할 문자열로 변환합니다."""

    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        ).strip()
    return str(content).strip()


def answer_question(
    question: str,
    vector_store: InMemoryVectorStore,
    llm: ChatOpenAI,
    chat_history: list[dict[str, str]],
) -> tuple[str, list[tuple[Document, float]]]:
    """질문과 관련된 문서를 검색하고 최신 LangChain 방식으로 답변을 생성합니다."""

    # 질문과 가까운 문서 조각만 검색합니다.
    # 표나 조건이 여러 청크에 걸쳐 있을 수 있으므로 검색 범위를 넓힙니다.
    results = vector_store.similarity_search_with_score(question, k=6)
    context = "\n\n".join(
        f"[청크: {doc.metadata.get('chunk_id', 'unknown')} | "
        f"출처: {doc.metadata['source']} / {doc.metadata['page']}쪽]\n"
        f"{doc.page_content}"
        for doc, _score in results
    )
    history_text = "\n".join(
        f"{message['role']}: {message['content']}" for message in chat_history[-6:]
    )

    # 구버전 RetrievalQA나 ConversationChain 대신 프롬프트와 llm.invoke를 직접 사용합니다.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """당신은 제공된 행정 문서만 근거로 답하는 한국어 안내자입니다.
다음 규칙을 반드시 지키세요.
1. 참고 문서에 직접 쓰인 사실만 답하고, 문서에 없는 숫자·조건·운임은 추측하거나 계산해 만들지 마세요.
2. 금액 질문은 금액, 지급 조건, 계산에 사용한 기간·등급·교통수단을 구분해서 답하세요.
3. 표 내용은 같은 행과 열의 항목을 섞지 말고, 표 제목과 단위를 함께 확인하세요.
4. 관련 근거가 여러 청크에 나뉘어 있으면 함께 비교한 뒤 답하세요. 서로 충돌하면 충돌 사실과 각각의 출처를 밝혀야 합니다.
5. 답변에 필요한 근거가 없으면 반드시 '문서에서 확인할 수 없습니다.'라고 답하고, 추가로 필요한 조건을 짧게 질문하세요.
6. 국내 지역 출장에 관한 질문은 금액을 항상 한화(원) 기준으로 안내하세요. 달러·외화·외국 국가나 불필요한 통화 설명은 언급하지 마세요.
7. 답변은 결론, 계산 또는 조건, 주의사항 순서로 짧고 명확하게 작성하세요.

참고 문서:
{context}""",
            ),
            (
                "human",
                "이전 대화:\n{history}\n\n현재 질문: {question}",
            ),
        ]
    )
    messages = prompt.invoke(
        {
            "context": context,
            "history": history_text or "(이전 대화 없음)",
            "question": question,
        }
    )
    response = llm.invoke(messages)
    return response_text(response), results


def main() -> None:
    st.set_page_config(page_title="공무원 여비 RAG 챗봇", page_icon="📚")
    st.title("📚 공무원 여비 문서 RAG 챗봇")
    st.caption("DATA 폴더의 문서만 근거로 답변합니다.")

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        st.error(".env 파일의 OPENAI_API_KEY에 OpenAI API 키를 입력하세요.")
        st.code("OPENAI_API_KEY=sk-...", language="dotenv")
        st.stop()

    try:
        vector_store, page_count, chunk_count = build_vector_store(api_key)
    except Exception as exc:
        st.error("문서 인덱스를 만들지 못했습니다. API 키와 네트워크를 확인하세요.")
        st.exception(exc)
        st.stop()

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
    st.info(f"문서 {page_count}페이지, 검색 조각 {chunk_count}개를 준비했습니다.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Streamlit은 화면이 다시 그려져도 session_state를 유지합니다.
    # 사용자가 명시적으로 버튼을 눌렀을 때만 대화 내용을 비웁니다.
    with st.sidebar:
        st.header("대화 관리")
        if st.button("대화 초기화", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("출처와 근거 문장"):
                    for source in message["sources"]:
                        st.markdown(source)

    question = st.chat_input("질문 입력")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    history = [
        {"role": message["role"], "content": message["content"]}
        for message in st.session_state.messages[:-1]
        if message["role"] in {"user", "assistant"}
    ]

    with st.chat_message("assistant"):
        with st.spinner("문서를 검색하고 답변을 만드는 중입니다..."):
            try:
                answer, results = answer_question(question, vector_store, llm, history)
            except Exception as exc:
                st.error("답변을 만드는 중 오류가 발생했습니다. API 키와 네트워크를 확인하세요.")
                st.exception(exc)
                return

        st.markdown(answer)
        source_lines = []
        seen_sources: set[tuple[str, int]] = set()
        for document, _score in results:
            source_key = (document.metadata["source"], document.metadata["page"])
            if source_key in seen_sources:
                continue
            seen_sources.add(source_key)
            evidence = make_evidence_sentence(document.page_content)
            source_lines.append(
                f"- **{source_key[0]}** ({source_key[1]}쪽)\n  - 근거 문장: {evidence}"
            )

        st.markdown("**출처 파일명과 근거 문장**")
        st.markdown("\n".join(source_lines))

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": source_lines,
        }
    )


if __name__ == "__main__":
    main()
