import os

from dotenv import load_dotenv

from rag_core import build_or_load_vectorstore, answer_question


def main() -> None:
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY. Put it in your .env file.")

    if not os.path.exists("resume.pdf"):
        raise RuntimeError("Missing resume.pdf in this folder.")

    vectorstore = build_or_load_vectorstore("resume.pdf")

    print("Resume RAG chat (OpenAI CLI). Type a question, or 'exit' to quit.\n")
    while True:
        q = input("> ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        try:
            answer, _docs = answer_question(vectorstore, q)
            print(f"\n{answer}\n")
        except Exception as e:
            print(f"\nError: {e}\n")
            break


if __name__ == "__main__":
    main()