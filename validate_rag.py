"""청크별 RAG 품질 검증 도구.

각 청크에서 질문과 기준 답변을 만들고, 실제 검색·답변 과정을 거친 뒤
검색된 원문과 답변이 일치하는지 LLM 평가자에게 확인받습니다.

기본 실행은 10개 청크만 검사합니다. 전체 검증은 다음처럼 실행합니다.

    uv run python validate_rag.py --limit 0

청크마다 질문 생성, 답변 생성, 평가까지 최대 3회의 모델 호출이 발생하므로
전체 214개 청크 검증은 API 비용과 시간이 발생할 수 있습니다.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field

from app import PROJECT_ROOT, read_pdf_documents, split_documents


class GeneratedQA(BaseModel):
    """원문 청크에서 생성한 질문과 기준 답변입니다."""

    question: str = Field(description="청크의 사실만 확인하는 한국어 질문")
    reference_answer: str = Field(description="청크만 근거로 작성한 기준 답변")


class JudgeResult(BaseModel):
    """검색·답변 결과에 대한 구조화된 평가입니다."""

    grounded: bool = Field(description="답변이 검색된 문서에 근거하는지")
    answer_matches_reference: bool = Field(
        description="답변이 기준 답변의 핵심 사실과 일치하는지"
    )
    reason: str = Field(description="판정 이유를 한국어로 설명")


def response_text(response: object) -> str:
    """LangChain 메시지 응답을 문자열로 바꿉니다."""

    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        ).strip()
    return str(content).strip()


def chunk_label(document: Document) -> str:
    """검증 결과에 표시할 청크 식별자입니다."""

    return document.metadata.get("chunk_id", "unknown")


def build_context(documents: list[Document]) -> str:
    """검색된 문서 조각을 평가용 문맥으로 합칩니다."""

    return "\n\n".join(
        f"[청크: {chunk_label(document)} | 출처: {document.metadata['source']} / "
        f"{document.metadata['page']}쪽]\n{document.page_content}"
        for document in documents
    )


def generate_qa(document: Document, llm: ChatOpenAI) -> GeneratedQA:
    """한 청크만 근거로 답할 수 있는 질문과 기준 답변을 만듭니다."""

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """당신은 RAG 평가용 질문 출제자입니다.
제공된 문서 청크에 직접 쓰인 사실만 사용해 질문 1개와 기준 답변 1개를 만드세요.
문서에 없는 내용을 보충하거나 추측하지 마세요.
질문은 금액·조건·절차·정의 중 하나를 확인할 수 있게 구체적으로 작성하세요.
기준 답변은 청크의 핵심 사실을 빠짐없이 포함하되 짧게 작성하세요.
금액은 원문에 단위가 있으면 그대로 유지하세요.

문서 청크:
{chunk}""",
            ),
            ("human", "위 청크로 평가용 QA를 생성하세요."),
        ]
    )
    structured_llm = llm.with_structured_output(GeneratedQA)
    return structured_llm.invoke(prompt.invoke({"chunk": document.page_content}))


def answer_from_context(question: str, context: str, llm: ChatOpenAI) -> str:
    """검색 문맥만 사용해 질문에 답합니다."""

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """당신은 RAG 답변 검증 대상입니다.
검색 문서에 직접 있는 내용만 답하세요.
검색 문서에 없는 내용은 추측하지 말고 '문서에서 확인할 수 없습니다.'라고 답하세요.
국내 출장 금액은 원화 기준으로만 답하세요.

검색 문서:
{context}""",
            ),
            ("human", "질문: {question}"),
        ]
    )
    return response_text(llm.invoke(prompt.invoke({"context": context, "question": question})))


def judge_answer(
    question: str,
    reference_answer: str,
    generated_answer: str,
    context: str,
    llm: ChatOpenAI,
) -> JudgeResult:
    """답변이 검색 문서에 근거하고 기준 답변과 맞는지 평가합니다."""

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """당신은 RAG 품질 평가자입니다.
다음 네 가지를 비교해 평가하세요.
- 질문
- 원문 청크에서 만든 기준 답변
- 검색 결과를 근거로 만든 실제 답변
- 실제 답변에 제공된 검색 문서

grounded는 실제 답변의 핵심 내용이 검색 문서에서 확인되면 true입니다.
answer_matches_reference는 실제 답변이 기준 답변의 핵심 사실과 일치하면 true입니다.
문서에 없는 숫자나 조건을 추가했으면 grounded를 false로 판정하세요.
답변이 '확인할 수 없습니다'라고 했지만 기준 답변을 만들 수 있는 근거가 있으면
answer_matches_reference를 false로 판정하세요.

질문:
{question}

기준 답변:
{reference_answer}

실제 답변:
{generated_answer}

검색 문서:
{context}""",
            ),
            ("human", "위 답변을 엄격하게 평가하세요."),
        ]
    )
    structured_llm = llm.with_structured_output(JudgeResult)
    return structured_llm.invoke(
        prompt.invoke(
            {
                "question": question,
                "reference_answer": reference_answer,
                "generated_answer": generated_answer,
                "context": context,
            }
        )
    )


def validate(limit: int, output_path: Path) -> None:
    """청크별 검증을 실행하고 JSON 보고서를 저장합니다."""

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(".env의 OPENAI_API_KEY가 비어 있습니다.")

    documents = split_documents(read_pdf_documents())
    selected_documents = documents if limit == 0 else documents[:limit]
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    vector_store = InMemoryVectorStore(embedding=embeddings)
    vector_store.add_documents(documents)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)

    records: list[dict[str, object]] = []
    for number, target_document in enumerate(selected_documents, start=1):
        print(f"[{number}/{len(selected_documents)}] {chunk_label(target_document)}")
        try:
            qa = generate_qa(target_document, llm)
            retrieved = vector_store.similarity_search_with_score(qa.question, k=6)
            retrieved_documents = [document for document, _score in retrieved]
            retrieved_ids = [chunk_label(document) for document in retrieved_documents]
            retrieval_hit = chunk_label(target_document) in retrieved_ids
            context = build_context(retrieved_documents)
            generated_answer = answer_from_context(qa.question, context, llm)
            judge = judge_answer(
                qa.question,
                qa.reference_answer,
                generated_answer,
                context,
                llm,
            )
            passed = retrieval_hit and judge.grounded and judge.answer_matches_reference
            records.append(
                {
                    "chunk_id": chunk_label(target_document),
                    "source": target_document.metadata["source"],
                    "page": target_document.metadata["page"],
                    "question": qa.question,
                    "reference_answer": qa.reference_answer,
                    "generated_answer": generated_answer,
                    "retrieved_chunk_ids": retrieved_ids,
                    "retrieval_hit": retrieval_hit,
                    "grounded": judge.grounded,
                    "answer_matches_reference": judge.answer_matches_reference,
                    "passed": passed,
                    "reason": judge.reason,
                }
            )
        except Exception as exc:
            records.append(
                {
                    "chunk_id": chunk_label(target_document),
                    "source": target_document.metadata["source"],
                    "page": target_document.metadata["page"],
                    "passed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    passed_count = sum(record.get("passed") is True for record in records)
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "total_chunks": len(documents),
        "validated_chunks": len(records),
        "passed": passed_count,
        "failed": len(records) - passed_count,
        "records": records,
    }
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"완료: {passed_count}/{len(records)} 통과")
    print(f"보고서: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 청크별 QA 검증")
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="검증할 청크 수. 0이면 전체 청크를 검증합니다. 기본값: 10",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "validation_report.json",
        help="검증 결과 JSON 경로",
    )
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit은 0 이상이어야 합니다.")
    validate(args.limit, args.output)


if __name__ == "__main__":
    main()
