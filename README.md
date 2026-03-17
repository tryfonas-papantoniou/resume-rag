# Interactive CV (RAG)

Two sections:

- **Chat with my CV**: RAG over `resume.pdf` (LangChain + Chroma + OpenAI)
- **About me**: static content from `ABOUT.md`

## Run locally (Windows PowerShell)

```powershell
cd "C:\Users\Tryfonas\Desktop\Cursor Projects\resume-rag"
.\.venv\Scripts\Activate.ps1
streamlit run app_streamlit.py
```

Create a `.env` file (do not share it) with:

```env
OPENAI_API_KEY=your_key_here
APP_PASSWORD=Accenture
```

## Deploy on Streamlit Community Cloud

1. Create a GitHub repo and push this folder (do **not** commit `.env`).
2. Go to Streamlit Community Cloud → **New app** → pick your repo.
3. Set:
   - **Main file path**: `app_streamlit.py`
4. In the app’s **Settings → Secrets**, add:

```toml
OPENAI_API_KEY = "YOUR_OPENAI_KEY"
APP_PASSWORD = "Accenture"
```

5. Deploy. Share the Streamlit URL + password.

