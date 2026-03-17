import os
from typing import List, Tuple

from PyPDF2 import PdfReader

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "resume"


def load_resume_pdf(path: str) -> str:
    reader = PdfReader(path)
    parts: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            parts.append(text)
    return "\n\n".join(parts).strip()


def build_or_load_vectorstore(
    resume_pdf_path: str,
    collection_name: str = COLLECTION_NAME,
    persist_dir: str = PERSIST_DIR,
    embedding_model: str = "text-embedding-3-small",
) -> Chroma:
    resume_text = load_resume_pdf(resume_pdf_path)
    if not resume_text:
        raise RuntimeError(
            "Could not extract any text from resume.pdf "
            "(it might be a scanned/image-only PDF)."
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_text(resume_text)
    docs = [Document(page_content=c, metadata={"source": resume_pdf_path}) for c in chunks]

    embeddings = OpenAIEmbeddings(model=embedding_model)

    vs = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )

    if vs._collection.count() == 0:
        vs.add_documents(docs)

    return vs


def answer_question(
    vectorstore: Chroma,
    question: str,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    k: int = 4,
) -> Tuple[str, List[Document]]:
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    retrieved_docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in retrieved_docs).strip()

    if not context:
        return "I don't know based on the resume.", retrieved_docs

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You answer questions using ONLY the provided resume context. "
                "If the answer is not in the context, say: \"I don't know based on the resume.\" "
                "Do not use outside knowledge.",
            ),
            ("human", "Resume context:\n{context}\n\nQuestion:\n{question}"),
        ]
    )

    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
    )

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question}).strip()
    return answer, retrieved_docs