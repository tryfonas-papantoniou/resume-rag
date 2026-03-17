import os
import time
import tempfile
import hashlib
from typing import Optional, Any

import streamlit as st
from dotenv import load_dotenv

from rag_core import build_or_load_vectorstore, answer_question


@st.cache_resource
def get_vectorstore(cache_key: str, resume_path: str) -> Any:
    return build_or_load_vectorstore(resume_path, collection_name=cache_key)


def get_secret(name: str) -> Optional[str]:
    try:
        if name in st.secrets:
            value = st.secrets.get(name)
            return str(value) if value is not None else None
    except FileNotFoundError:
        pass
    return os.getenv(name)


def read_about_markdown() -> str:
    try:
        with open("ABOUT.md", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "## About me\n\nCreate an `ABOUT.md` file to customize this section.\n"


def show_home() -> None:
    # Full-height split layout with zig‑zag divider
    st.markdown(
        """
        <style>
        body {
            margin: 0 !important;
        }
        [data-testid="stAppViewContainer"] > .main {
            padding: 0;
            background: #05010f;
            color: #f9fafb;
        }
        [data-testid="stAppViewContainer"] .block-container {
            padding-top: 0;
            padding-bottom: 0;
            max-width: 100%;
        }

        .split-root {
            position: relative;
            width: 100vw;
            height: 100vh;
            overflow: hidden;
            display: flex;
        }

        .split-half {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 3rem 3.5rem;
            box-sizing: border-box;
        }

        .split-left {
            background: radial-gradient(circle at 0% 0%, #8e3bff 0%, #3b006b 40%, #1b0735 100%);
        }

        .split-right {
            background: radial-gradient(circle at 100% 0%, #ff66cc 0%, #5d1b7e 35%, #120524 100%);
        }

        .panel-inner {
            max-width: 460px;
            color: #f9fafb;
        }

        .pill {
            display: inline-flex;
            align-items: center;
            padding: 0.35rem 0.9rem;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.6);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            opacity: 0.9;
        }

        .title-row {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            margin-bottom: 0.5rem;
        }

        .title {
            font-size: 2.3rem;
            font-weight: 800;
        }

        .subtitle {
            font-size: 0.98rem;
            line-height: 1.6;
            opacity: 0.95;
            margin-bottom: 1.5rem;
        }

        .primary-link {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.65rem 1.3rem;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.9);
            color: #0b0618;
            background: #ffffff;
            font-weight: 600;
            font-size: 0.9rem;
            text-decoration: none;
        }

        .primary-link span.arrow {
            font-size: 1rem;
        }

        .primary-link:hover {
            background: #f3f4ff;
        }

        /* center zig‑zag divider */
        .divider-wrap {
            position: absolute;
            inset: 0;
            pointer-events: none;
            display: flex;
            align-items: stretch;
            justify-content: center;
        }

        .divider-inner {
            position: relative;
            width: 84px;
            max-height: 100%;
        }

        .divider-inner svg {
            position: absolute;
            top: -40px;
            bottom: -40px;
            left: 0;
            right: 0;
            width: 100%;
            height: calc(100% + 80px);
        }
        </style>

        <div class="split-root">
          <div class="split-half split-left">
            <div class="panel-inner">
              <div class="title-row">
                <div class="title">About Me</div>
                <div class="pill">Recommended 1st</div>
              </div>
              <div class="subtitle">
                A short, curated overview: why I’m pivoting to GenAI, what I built,
                and what I’m learning.
              </div>
              <a class="primary-link" href="?view=about">
                <span>Open About Me</span>
                <span class="arrow">→</span>
              </a>
            </div>
          </div>

          <div class="split-half split-right">
            <div class="panel-inner">
              <div class="title-row">
                <div class="title">CV Chat (RAG)</div>
                <div class="pill">Interactive</div>
              </div>
              <div class="subtitle">
                Ask questions about my resume and get answers grounded in the PDF,
                with snippets shown. Or upload your own CV.
              </div>
              <a class="primary-link" href="?view=chat">
                <span>Open CV Chat</span>
                <span class="arrow">→</span>
              </a>
            </div>
          </div>

          <div class="divider-wrap" aria-hidden="true">
            <div class="divider-inner">
              <svg viewBox="0 0 80 800" preserveAspectRatio="none">
                <defs>
                  <pattern id="zip-zag" width="80" height="120" patternUnits="userSpaceOnUse">
                    <polyline
                      points="40,0 26,22 52,44 28,66 54,88 32,110 40,120"
                      fill="none"
                      stroke="#ffffff"
                      stroke-width="6"
                      stroke-linecap="round"
                      stroke-linejoin="round"
                      opacity="0.97"
                    />
                    <polyline
                      points="46,0 34,22 58,44 36,66 60,88 40,110 46,120"
                      fill="none"
                      stroke="#e5d9ff"
                      stroke-width="2.6"
                      stroke-linecap="round"
                      stroke-linejoin="round"
                      opacity="0.9"
                    />
                  </pattern>
                </defs>
                <rect x="0" y="0" width="80" height="800" fill="url(#zip-zag)" />
              </svg>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_about() -> None:
    if st.button("← Back to home", key="btn_back_about"):
        st.query_params["view"] = "home"
        st.rerun()
    st.title("🧠 About me")
    st.markdown(read_about_markdown())


def show_chat() -> None:
    if st.button("← Back to home", key="btn_back_chat"):
        st.query_params["view"] = "home"
        st.rerun()
    st.title("📄 Chat with My Resume")
    st.write(
        "Ask my definitely-not-fake resume questions about me! Or upload your own resume!\n\n"
        "Answers are based **only** on the resume content."
    )
    st.markdown("**Sample questions:**")
    st.markdown(
        "- \"Summarize my responsibilities in my latest role.\"\n"
        "- \"What is the favorite cooking recipe of the applicant?\"\n"
        "- \"Is this applicant Batman?\""
    )

    with st.sidebar:
        st.header("Settings")

        model_name = st.selectbox(
            "OpenAI chat model",
            options=["gpt-4o-mini", "gpt-4o"],
            index=0,
        )

        temperature = st.slider(
            "Creativity (temperature)",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
        )

        k = st.slider(
            "Number of text chunks to retrieve (k)",
            min_value=2,
            max_value=10,
            value=4,
            step=1,
        )

        st.caption(f"Next question will use: temperature = {temperature}, k = {k}")

        st.markdown("---")
        st.subheader("Upload resume (optional)")
        uploaded_file = st.file_uploader(
            "Upload a PDF resume to use instead of the default `resume.pdf`.",
            type=["pdf"],
        )

    resume_path: Optional[str] = None
    cache_key = "resume_default"

    if uploaded_file is not None:
        raw = uploaded_file.getvalue()
        file_hash = hashlib.sha256(raw).hexdigest()[:16]
        tmp_dir = tempfile.gettempdir()
        resume_path = os.path.join(tmp_dir, f"uploaded_resume_{file_hash}.pdf")
        with open(resume_path, "wb") as f:
            f.write(raw)
        cache_key = f"resume_uploaded_{file_hash}"
    else:
        if not os.path.exists("resume.pdf"):
            st.error("No `resume.pdf` found in this folder, and no upload provided.")
            st.stop()
        resume_path = "resume.pdf"
        # Cache key includes file content hash so replacing resume.pdf picks up the new content
        with open(resume_path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()[:16]
        cache_key = f"resume_default_{file_hash}"

    try:
        vectorstore = get_vectorstore(cache_key, resume_path)
    except Exception as e:
        st.error(f"Error building search index: {e}")
        st.stop()

    if "last_ask_ts" not in st.session_state:
        st.session_state.last_ask_ts = 0.0

    if "history" not in st.session_state:
        st.session_state.history = []

    # Center "Question" label and align Ask button with the input field
    st.markdown(
        """
        <style>
        /* Align the form row so the Ask button sits at same height as the input */
        [data-testid="stForm"] > div [data-testid="stHorizontalBlock"] {
            align-items: flex-end;
        }
        /* Center the custom Question label */
        .question-label-center {
            text-align: center;
            margin-bottom: 0.25rem;
            font-weight: 600;
            font-size: 0.875rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # Form so that pressing Enter in the text box submits the question
    with st.form(key="question_form"):
        col_q, col_btn = st.columns([7, 2])
        with col_q:
            st.markdown('<p class="question-label-center">Question</p>', unsafe_allow_html=True)
            question = st.text_input(
                label="Question",
                placeholder='e.g. "Summarize my professional experience"',
                label_visibility="collapsed",
            )
        with col_btn:
            submitted = st.form_submit_button("Ask", use_container_width=True)

    if submitted and question.strip():
        now = time.time()
        if now - st.session_state.last_ask_ts < 3.0:
            st.warning("You are asking too quickly. Please wait at least 3 seconds between questions.")
        else:
            st.session_state.last_ask_ts = now
            try:
                with st.spinner("Thinking..."):
                    answer, docs = answer_question(
                        vectorstore,
                        question,
                        model_name=model_name,
                        temperature=temperature,
                        k=k,
                    )
                st.session_state.history.append(
                    {
                        "question": question,
                        "answer": answer,
                        "sources": [d.page_content for d in docs],
                    }
                )
            except Exception as e:
                st.error(f"Error getting answer: {e}")

    for turn in reversed(st.session_state.history):
        st.markdown(f"**You:** {turn['question']}")
        st.markdown(f"**Your CV answered:** {turn['answer']}")
        with st.expander("Show resume snippets used for this answer"):
            for i, snippet in enumerate(turn["sources"], start=1):
                st.markdown(f"**Snippet {i}:**")
                st.code(snippet[:1200])
        st.markdown("---")

    st.markdown(
        "<p style='font-size: 0.7rem; color: #666; margin-top: 1.5rem;'>"
         "*** The CV I use includes comedic elements and is not my original one - "
        "Feel free to ask me for my real CV and it will be provided.</p>",
        unsafe_allow_html=True,
    )


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="Interactive CV", page_icon="📄", layout="centered")

    # Accenture purple for button hover (instead of default red)
    st.markdown(
        """
        <style>
        .stButton > button:hover, .stButton > button:focus {
            border-color: rgb(90, 0, 107) !important;
            background-color: rgb(90, 0, 107) !important;
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if not get_secret("OPENAI_API_KEY"):
        st.error("Missing OPENAI_API_KEY. Set it in `.env` or Streamlit Secrets.")
        st.stop()

    view = st.query_params.get("view") or "home"

    if view == "about":
        show_about()
    elif view == "chat":
        show_chat()
    else:
        show_home()


if __name__ == "__main__":
    main()

