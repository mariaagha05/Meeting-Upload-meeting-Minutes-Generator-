"""
AI Meeting Minutes Generator
----------------------------
Upload meeting notes (TXT / MD / DOCX / PDF) or a recording (MP3 / WAV / M4A / MP4 ...),
or paste text. The app transcribes audio (Groq Whisper), then uses a Groq LLM to produce:
summary, key topics, decisions, action items (owner, deadline, priority) and open questions.
Everything is editable and exportable (DOCX, Markdown, CSV, JSON).

Run:  streamlit run app.py
"""

import io
import json
import os
import re
import time
from datetime import date

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Meeting Minutes AI",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ask Streamlit for the light theme so text/table colours always match the CSS below.
try:
    from streamlit import config as _st_config

    _st_config.set_option("theme.base", "light")
except Exception:
    pass

# Optional libraries are imported safely so the app never crashes on start-up.
try:
    from groq import Groq
    import groq as groq_errors
except Exception:  # pragma: no cover
    Groq = None
    groq_errors = None

try:
    from docx import Document
    from docx.shared import Pt
except Exception:  # pragma: no cover
    Document = None

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    PdfReader = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
AUDIO_EXT = {"mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm", "ogg", "flac"}
TEXT_EXT = {"txt", "md", "docx", "pdf"}
MAX_AUDIO_MB = 25  # Groq free-tier upload limit
CHUNK_CHARS = 16000  # keeps each LLM request small and fast

LLM_MODELS = {
    "Llama 3.3 70B (best quality)": "llama-3.3-70b-versatile",
    "Llama 3.1 8B (fastest)": "llama-3.1-8b-instant",
}
WHISPER_MODELS = {
    "Whisper Large v3 Turbo (fast)": "whisper-large-v3-turbo",
    "Whisper Large v3 (most accurate)": "whisper-large-v3",
}
LANGUAGES = {
    "Auto-detect": None,
    "English": "en",
    "Urdu": "ur",
    "Hindi": "hi",
    "Arabic": "ar",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
}
PRIORITIES = ["High", "Medium", "Low"]
STATUSES = ["Open", "In progress", "Done"]


class AppError(Exception):
    """Friendly, user-facing error."""


# ---------------------------------------------------------------------------
# Styling  (high-contrast, readable fonts, balanced layout)
# ---------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
  --ink: #1B2430;
  --ink-soft: #46546A;
  --bg: #F5F7FB;
  --card: #FFFFFF;
  --line: #D9E0EC;
  --accent: #2F5BEA;
  --accent-dark: #2447BD;
  --accent-tint: #E8EEFF;
  --good: #167A4A;
  --warn: #B45309;
}

html, body, .stApp, [class*="css"] {
  font-family: 'Plus Jakarta Sans', 'Segoe UI', Arial, sans-serif !important;
}
.stApp { background: var(--bg) !important; color: var(--ink) !important; }
.block-container { padding-top: 1.6rem !important; max-width: 1180px; }

/* Text colours: always dark on light */
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5 { color: var(--ink) !important; font-weight: 700 !important; letter-spacing: -0.01em; }
.stApp p, .stApp li, .stApp label, .stApp .stMarkdown, .stApp [data-testid="stWidgetLabel"] p,
.stApp [data-testid="stCaptionContainer"] { color: var(--ink) !important; font-size: 1rem; line-height: 1.6; }
.stApp [data-testid="stCaptionContainer"] p { color: var(--ink-soft) !important; font-size: .92rem; }

/* Sidebar */
[data-testid="stSidebar"] { background: #FFFFFF !important; border-right: 1px solid var(--line); }
[data-testid="stSidebar"] * { color: var(--ink) !important; }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { font-size: 1.1rem !important; }

/* Hero */
.hero {
  background: linear-gradient(135deg, #1E3A8A 0%, #2F5BEA 100%);
  border-radius: 16px; padding: 28px 32px; margin-bottom: 22px;
}
.hero h1 { color: #FFFFFF !important; font-size: 2rem !important; margin: 0 0 6px 0 !important; }
.hero p  { color: #E6ECFF !important; font-size: 1.05rem !important; margin: 0 !important; }

/* Cards */
.card {
  background: var(--card); border: 1px solid var(--line); border-radius: 14px;
  padding: 18px 20px; margin-bottom: 14px;
}
.card h3 { margin: 0 0 4px 0 !important; font-size: 1.1rem !important; }
.card p  { margin: 0 !important; color: var(--ink-soft) !important; }

/* Metrics */
.metric-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 6px 0 18px 0; }
.metric { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px; }
.metric .num { font-size: 1.8rem; font-weight: 800; color: var(--accent-dark); line-height: 1.1; }
.metric .lbl { font-size: .9rem; color: var(--ink-soft); font-weight: 500; }
@media (max-width: 800px) { .metric-row { grid-template-columns: repeat(2, 1fr); } }

/* Inputs */
.stTextInput input, .stTextArea textarea, .stDateInput input, .stNumberInput input {
  background: #FFFFFF !important; color: var(--ink) !important;
  border: 1.5px solid var(--line) !important; border-radius: 10px !important; font-size: 1rem !important;
}
.stTextInput input:focus, .stTextArea textarea:focus { border-color: var(--accent) !important; box-shadow: 0 0 0 3px var(--accent-tint) !important; }
div[data-baseweb="select"] > div { background: #FFFFFF !important; color: var(--ink) !important; border: 1.5px solid var(--line) !important; border-radius: 10px !important; }
div[data-baseweb="select"] * { color: var(--ink) !important; }
div[data-baseweb="popover"] * { color: var(--ink) !important; background: #FFFFFF; }

/* File uploader */
[data-testid="stFileUploaderDropzone"] { background: #FFFFFF !important; border: 2px dashed #9DB1F5 !important; border-radius: 12px !important; }
[data-testid="stFileUploaderDropzone"] * { color: var(--ink) !important; }
[data-testid="stFileUploaderDropzone"] button { background: var(--accent-tint) !important; border: 1px solid var(--accent) !important; }

/* Buttons */
.stButton > button, .stDownloadButton > button {
  border-radius: 10px !important; font-weight: 600 !important; font-size: 1rem !important;
  padding: .6rem 1.2rem !important; border: 1.5px solid var(--accent) !important;
  background: #FFFFFF !important; color: var(--accent-dark) !important;
}
.stButton > button p, .stDownloadButton > button p { color: inherit !important; }
.stButton > button:hover, .stDownloadButton > button:hover { background: var(--accent-tint) !important; }
.stButton > button[kind="primary"] { background: var(--accent) !important; color: #FFFFFF !important; border-color: var(--accent) !important; }
.stButton > button[kind="primary"]:hover { background: var(--accent-dark) !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { font-weight: 600; font-size: 1rem; color: var(--ink-soft) !important; padding: 10px 16px; }
.stTabs [aria-selected="true"] { color: var(--accent-dark) !important; }
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--accent) !important; }

/* Tables / alerts / expander */
[data-testid="stDataFrame"], [data-testid="stDataEditor"] { border: 1px solid var(--line); border-radius: 10px; }
[data-testid="stAlert"] * { color: var(--ink) !important; }
[data-testid="stExpander"] { background: #FFFFFF; border: 1px solid var(--line) !important; border-radius: 12px; }
[data-testid="stExpander"] summary p { color: var(--ink) !important; font-weight: 600; }
#MainMenu, footer { visibility: hidden; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers: API key, client, file reading
# ---------------------------------------------------------------------------
def get_default_key() -> str:
    try:
        if "GROQ_API_KEY" in st.secrets:
            return str(st.secrets["GROQ_API_KEY"])
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY", "")


def get_client(api_key: str):
    if Groq is None:
        raise AppError("The 'groq' package is not installed. Run: pip install -r requirements.txt")
    if not api_key or not api_key.strip():
        raise AppError("Please paste your Groq API key in the sidebar first.")
    return Groq(api_key=api_key.strip())


def explain_error(exc: Exception) -> str:
    """Turn any exception into a clear, human message."""
    if isinstance(exc, AppError):
        return str(exc)
    name = type(exc).__name__
    text = str(exc)
    low = text.lower()
    if name == "AuthenticationError" or "invalid api key" in low or "401" in low:
        return "Your Groq API key was rejected. Check the key in the sidebar (it should start with 'gsk_')."
    if name == "RateLimitError" or "rate limit" in low or "429" in low:
        return "Groq's rate limit was reached. Wait about a minute and try again, or choose the faster 8B model."
    if name in ("APIConnectionError", "APITimeoutError") or "connection" in low or "timed out" in low:
        return "Could not reach Groq. Check your internet connection and try again."
    if "model" in low and ("not found" in low or "decommissioned" in low or "does not exist" in low):
        return "The selected model is not available on your Groq account. Pick another model in the sidebar."
    if "413" in low or "too large" in low or "request entity" in low:
        return "The file or text is too large for Groq. Try a shorter recording or smaller document."
    return f"Something went wrong: {text[:300]}"


def read_text_file(name: str, data: bytes) -> str:
    ext = name.rsplit(".", 1)[-1].lower()
    if ext in ("txt", "md"):
        for enc in ("utf-8", "utf-16", "latin-1"):
            try:
                return data.decode(enc)
            except Exception:
                continue
        return data.decode("utf-8", errors="replace")
    if ext == "docx":
        if Document is None:
            raise AppError("DOCX support needs 'python-docx'. Run: pip install -r requirements.txt")
        doc = Document(io.BytesIO(data))
        parts = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))
        return "\n".join(parts)
    if ext == "pdf":
        if PdfReader is None:
            raise AppError("PDF support needs 'pypdf'. Run: pip install -r requirements.txt")
        reader = PdfReader(io.BytesIO(data))
        pages = [(p.extract_text() or "") for p in reader.pages]
        text = "\n".join(pages).strip()
        if not text:
            raise AppError("This PDF has no readable text (it may be a scan). Upload a text PDF or paste the text.")
        return text
    raise AppError(f"Unsupported file type: .{ext}")


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------
def transcribe_audio(client, name: str, data: bytes, model: str, language) -> str:
    size_mb = len(data) / (1024 * 1024)
    if size_mb > MAX_AUDIO_MB:
        raise AppError(
            f"This recording is {size_mb:.1f} MB. Groq accepts up to {MAX_AUDIO_MB} MB per file. "
            "Compress it (e.g. export as a mono 64 kbps MP3) or split it into parts."
        )
    kwargs = {
        "file": (name, data),
        "model": model,
        "response_format": "verbose_json",
        "temperature": 0.0,
    }
    if language:
        kwargs["language"] = language
    resp = client.audio.transcriptions.create(**kwargs)
    text = getattr(resp, "text", None)
    if text is None and isinstance(resp, dict):
        text = resp.get("text")
    if text is None:
        text = str(resp)
    text = text.strip()
    if not text:
        raise AppError("No speech was detected in this recording. Check that the audio is audible.")
    return text


# ---------------------------------------------------------------------------
# LLM extraction
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert meeting secretary. You read a meeting transcript or notes and produce precise, factual minutes.
Rules:
- Use ONLY information present in the text. Never invent people, decisions, tasks or dates.
- If an owner is not stated, use "Unassigned". If no deadline is stated, use "Not specified".
- If a deadline is relative (e.g. "by Friday", "next week") and a meeting date is provided, convert it to an
  ISO date (YYYY-MM-DD) when it can be done reliably, and otherwise keep the original wording.
- Decisions are things the group agreed or resolved. Action items are tasks someone must do.
- Priority is High, Medium or Low (High only if urgency is stated or clearly implied).
- Write in the same language as the meeting text.
- Respond with ONE valid JSON object only, no markdown, matching exactly this schema:
{
  "title": "short meeting title",
  "attendees": ["name", ...],
  "summary": "3-6 sentence executive summary",
  "key_topics": ["topic", ...],
  "decisions": [{"decision": "text", "context": "why / details"}],
  "action_items": [{"task": "text", "owner": "name or Unassigned", "deadline": "date or Not specified", "priority": "High|Medium|Low"}],
  "open_questions": ["question", ...]
}"""


def split_chunks(text: str, size: int = CHUNK_CHARS):
    text = text.strip()
    if len(text) <= size:
        return [text]
    chunks, current = [], ""
    for line in text.splitlines():
        while len(line) > size:  # extremely long line: hard split
            if current:
                chunks.append(current)
                current = ""
            chunks.append(line[:size])
            line = line[size:]
        if len(current) + len(line) + 1 > size and current:
            chunks.append(current)
            current = ""
        current += line + "\n"
    if current.strip():
        chunks.append(current)
    return chunks


def parse_json(raw: str) -> dict:
    raw = (raw or "").strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except Exception:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except Exception:
                pass
    raise AppError("The AI returned an unreadable answer. Please click Generate again.")


def call_llm(client, model: str, user_prompt: str) -> dict:
    last_exc = None
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=3000,
                response_format={"type": "json_object"},
            )
            return parse_json(resp.choices[0].message.content)
        except AppError as exc:  # bad JSON -> retry once or twice
            last_exc = exc
        except Exception as exc:
            last_exc = exc
            if "rate" in str(exc).lower() or "429" in str(exc):
                time.sleep(6 * (attempt + 1))
            elif attempt == 2:
                raise
    raise last_exc if last_exc else AppError("Unknown AI error.")


def extract_minutes(client, model: str, text: str, meeting_date: date) -> dict:
    chunks = split_chunks(text)
    header = f"Meeting date: {meeting_date.isoformat()} ({meeting_date.strftime('%A')})\n\n"
    if len(chunks) == 1:
        return call_llm(client, model, header + "Meeting text:\n" + chunks[0])

    partials = []
    for i, chunk in enumerate(chunks, 1):
        prompt = header + f"This is part {i} of {len(chunks)} of a long meeting.\n\nMeeting text:\n{chunk}"
        partials.append(call_llm(client, model, prompt))
    merge_prompt = (
        header
        + "Below are JSON minutes for consecutive parts of ONE meeting. Merge them into a single set of minutes. "
        "Remove duplicates, keep every distinct decision and action item, and write one coherent summary.\n\n"
        + json.dumps(partials, ensure_ascii=False)
    )
    return call_llm(client, model, merge_prompt)


# ---------------------------------------------------------------------------
# Normalisation + data frames
# ---------------------------------------------------------------------------
def as_list(value):
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def normalise(data: dict) -> dict:
    decisions = []
    for d in data.get("decisions") or []:
        if isinstance(d, dict):
            decisions.append({"Decision": str(d.get("decision", "")).strip(), "Context": str(d.get("context", "")).strip()})
        elif isinstance(d, str):
            decisions.append({"Decision": d.strip(), "Context": ""})
    actions = []
    for a in data.get("action_items") or []:
        if isinstance(a, dict):
            pr = str(a.get("priority", "Medium")).strip().capitalize()
            actions.append(
                {
                    "Task": str(a.get("task", "")).strip(),
                    "Owner": str(a.get("owner", "Unassigned")).strip() or "Unassigned",
                    "Deadline": str(a.get("deadline", "Not specified")).strip() or "Not specified",
                    "Priority": pr if pr in PRIORITIES else "Medium",
                    "Status": "Open",
                }
            )
        elif isinstance(a, str):
            actions.append({"Task": a.strip(), "Owner": "Unassigned", "Deadline": "Not specified", "Priority": "Medium", "Status": "Open"})
    return {
        "title": str(data.get("title") or "Meeting Minutes").strip(),
        "attendees": as_list(data.get("attendees")),
        "summary": str(data.get("summary") or "").strip(),
        "topics": as_list(data.get("key_topics")),
        "questions": as_list(data.get("open_questions")),
        "decisions": pd.DataFrame(decisions, columns=["Decision", "Context"]),
        "actions": pd.DataFrame(actions, columns=["Task", "Owner", "Deadline", "Priority", "Status"]),
    }


def clean_df(df: pd.DataFrame, required: str, defaults: dict) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy().fillna("")
    for col, default in defaults.items():
        if col in out.columns:
            out[col] = out[col].apply(lambda v: v if str(v).strip() else default)
    out = out[out[required].astype(str).str.strip() != ""]
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------
def build_markdown(m: dict) -> str:
    lines = [f"# {m['title']}", f"*Date: {m['date']}*", ""]
    if m["attendees"]:
        lines += ["**Attendees:** " + ", ".join(m["attendees"]), ""]
    lines += ["## Summary", m["summary"] or "-", ""]
    if m["topics"]:
        lines += ["## Key topics"] + [f"- {t}" for t in m["topics"]] + [""]
    lines += ["## Decisions"]
    if len(m["decisions"]):
        for _, r in m["decisions"].iterrows():
            ctx = f" — {r['Context']}" if str(r["Context"]).strip() else ""
            lines.append(f"- {r['Decision']}{ctx}")
    else:
        lines.append("- None recorded")
    lines += ["", "## Action items", "| Task | Owner | Deadline | Priority | Status |", "|---|---|---|---|---|"]
    for _, r in m["actions"].iterrows():
        cells = [str(r[c]).replace("|", "/") for c in ["Task", "Owner", "Deadline", "Priority", "Status"]]
        lines.append("| " + " | ".join(cells) + " |")
    if m["questions"]:
        lines += ["", "## Open questions"] + [f"- {q}" for q in m["questions"]]
    return "\n".join(lines) + "\n"


def build_docx(m: dict):
    if Document is None:
        return None
    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(11)
    doc.add_heading(m["title"], 0)
    doc.add_paragraph(f"Date: {m['date']}")
    if m["attendees"]:
        doc.add_paragraph("Attendees: " + ", ".join(m["attendees"]))
    doc.add_heading("Summary", 1)
    doc.add_paragraph(m["summary"] or "-")
    if m["topics"]:
        doc.add_heading("Key topics", 1)
        for t in m["topics"]:
            doc.add_paragraph(t, style="List Bullet")

    def add_table(headers, rows):
        table = doc.add_table(rows=1, cols=len(headers))
        table.style = "Table Grid"
        for i, h in enumerate(headers):
            cell = table.rows[0].cells[i]
            cell.text = ""
            cell.paragraphs[0].add_run(h).bold = True
        for row in rows:
            cells = table.add_row().cells
            for i, v in enumerate(row):
                cells[i].text = str(v)

    doc.add_heading("Decisions", 1)
    if len(m["decisions"]):
        add_table(["Decision", "Context"], m["decisions"][["Decision", "Context"]].values.tolist())
    else:
        doc.add_paragraph("None recorded")
    doc.add_heading("Action items", 1)
    if len(m["actions"]):
        add_table(["Task", "Owner", "Deadline", "Priority", "Status"], m["actions"].values.tolist())
    else:
        doc.add_paragraph("None recorded")
    if m["questions"]:
        doc.add_heading("Open questions", 1)
        for q in m["questions"]:
            doc.add_paragraph(q, style="List Bullet")
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()[:50] or "meeting_minutes"


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
for key, default in {"minutes": None, "transcript": "", "gen": 0, "meeting_date": date.today()}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def run_pipeline(client, llm_model, text, meeting_date):
    """Extract minutes from text and store them in session state."""
    with st.spinner("Extracting summary, decisions and action items..."):
        data = extract_minutes(client, llm_model, text, meeting_date)
    st.session_state.minutes = normalise(data)
    st.session_state.transcript = text
    st.session_state.meeting_date = meeting_date
    st.session_state.gen += 1


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## Settings")
    api_key = st.text_input(
        "Groq API key",
        value=get_default_key(),
        type="password",
        placeholder="gsk_...",
        help="Get a free key at console.groq.com. It is used only for this session and is never saved.",
    )
    llm_label = st.selectbox("Summary model", list(LLM_MODELS.keys()))
    whisper_label = st.selectbox("Transcription model", list(WHISPER_MODELS.keys()))
    lang_label = st.selectbox("Recording language", list(LANGUAGES.keys()))
    meeting_date_in = st.date_input("Meeting date", value=date.today(), help="Used to turn 'by Friday' into a real date.")
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown(
        "1. Add notes or a recording\n"
        "2. Click **Generate minutes**\n"
        "3. Edit anything in the tables\n"
        "4. Download as Word, Markdown or CSV"
    )
    st.caption(f"Recordings up to {MAX_AUDIO_MB} MB. Formats: MP3, WAV, M4A, MP4, WEBM, OGG, FLAC.")
    if st.button("Start over", use_container_width=True):
        st.session_state.minutes = None
        st.session_state.transcript = ""
        st.session_state.gen += 1
        st.rerun()

llm_model = LLM_MODELS[llm_label]
whisper_model = WHISPER_MODELS[whisper_label]
language = LANGUAGES[lang_label]

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
<div class="hero">
  <h1>Meeting Minutes AI</h1>
  <p>Turn a recording or rough notes into a clean summary, decisions and action items with owners and deadlines.</p>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Input section
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="card"><h3>1. Add your meeting</h3><p>Upload a file or paste text. Audio is transcribed automatically.</p></div>',
    unsafe_allow_html=True,
)

tab_upload, tab_paste = st.tabs(["Upload file", "Paste text"])
with tab_upload:
    uploaded = st.file_uploader(
        "Recording or notes",
        type=sorted(AUDIO_EXT | TEXT_EXT),
        accept_multiple_files=False,
        label_visibility="collapsed",
    )
    if uploaded is not None:
        kind = "recording" if uploaded.name.rsplit(".", 1)[-1].lower() in AUDIO_EXT else "notes"
        st.caption(f"Selected {kind}: {uploaded.name} ({uploaded.size / (1024 * 1024):.2f} MB)")
        if kind == "recording":
            try:
                st.audio(uploaded.getvalue())
            except Exception:
                pass
with tab_paste:
    pasted = st.text_area(
        "Meeting notes or transcript",
        height=220,
        placeholder="Paste your meeting notes or transcript here...",
        label_visibility="collapsed",
    )

go = st.button("Generate minutes", type="primary", use_container_width=True)

if go:
    try:
        client = get_client(api_key)
        source_text = ""
        if uploaded is not None:
            name = uploaded.name
            ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
            data = uploaded.getvalue()
            if ext in AUDIO_EXT:
                with st.spinner("Transcribing audio... longer recordings can take a minute."):
                    source_text = transcribe_audio(client, name, data, whisper_model, language)
            elif ext in TEXT_EXT:
                with st.spinner("Reading your document..."):
                    source_text = read_text_file(name, data)
            else:
                raise AppError("This file type is not supported. Use TXT, MD, DOCX, PDF or an audio file.")
        elif pasted.strip():
            source_text = pasted.strip()
        else:
            raise AppError("Add a file or paste some text first, then click Generate minutes.")

        if len(source_text.split()) < 15:
            raise AppError("The text is too short to create minutes. Add more detail and try again.")

        run_pipeline(client, llm_model, source_text, meeting_date_in)
        st.success("Minutes are ready. Review and edit them below.")
    except Exception as exc:  # never show a raw traceback
        st.error(explain_error(exc))

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
m = st.session_state.minutes
if m is None:
    st.info("Your minutes will appear here after you add a meeting and click Generate minutes.")
else:
    gen = st.session_state.gen
    st.markdown(
        '<div class="card"><h3>2. Review and edit</h3><p>Every field below is editable. Add or delete table rows as needed.</p></div>',
        unsafe_allow_html=True,
    )

    tab_sum, tab_dec, tab_act, tab_tr, tab_exp = st.tabs(
        ["Summary", "Decisions", "Action items", "Transcript", "Export"]
    )

    with tab_sum:
        title = st.text_input("Meeting title", value=m["title"], key=f"title_{gen}")
        attendees_txt = st.text_area(
            "Attendees (one per line)", value="\n".join(m["attendees"]), height=110, key=f"att_{gen}"
        )
        summary = st.text_area("Summary", value=m["summary"], height=200, key=f"sum_{gen}")
        topics_txt = st.text_area("Key topics (one per line)", value="\n".join(m["topics"]), height=130, key=f"top_{gen}")
        questions_txt = st.text_area(
            "Open questions (one per line)", value="\n".join(m["questions"]), height=110, key=f"q_{gen}"
        )

    with tab_dec:
        st.caption("Things the group agreed or resolved.")
        dec_df = st.data_editor(
            m["decisions"],
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            key=f"dec_{gen}",
            column_config={
                "Decision": st.column_config.TextColumn("Decision", width="large", required=False),
                "Context": st.column_config.TextColumn("Context", width="large"),
            },
        )

    with tab_act:
        st.caption("Tasks with owners and deadlines. Deadlines can be a date or plain text such as 'next Friday'.")
        act_df = st.data_editor(
            m["actions"],
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            key=f"act_{gen}",
            column_config={
                "Task": st.column_config.TextColumn("Task", width="large"),
                "Owner": st.column_config.TextColumn("Owner", width="small"),
                "Deadline": st.column_config.TextColumn("Deadline", width="small"),
                "Priority": st.column_config.SelectboxColumn("Priority", options=PRIORITIES, width="small"),
                "Status": st.column_config.SelectboxColumn("Status", options=STATUSES, width="small"),
            },
        )

    with tab_tr:
        st.caption("The source text. You can fix mistakes here and regenerate the minutes.")
        edited_transcript = st.text_area(
            "Transcript", value=st.session_state.transcript, height=360, key=f"tr_{gen}", label_visibility="collapsed"
        )
        if st.button("Regenerate minutes from this text"):
            try:
                client = get_client(api_key)
                run_pipeline(client, llm_model, edited_transcript, meeting_date_in)
                st.rerun()
            except Exception as exc:
                st.error(explain_error(exc))

    # Build the current (edited) snapshot for metrics + export
    final_dec = clean_df(dec_df, "Decision", {"Context": ""})
    final_act = clean_df(
        act_df, "Task", {"Owner": "Unassigned", "Deadline": "Not specified", "Priority": "Medium", "Status": "Open"}
    )
    snapshot = {
        "title": title.strip() or "Meeting Minutes",
        "date": st.session_state.meeting_date.isoformat(),
        "attendees": [a.strip() for a in attendees_txt.splitlines() if a.strip()],
        "summary": summary.strip(),
        "topics": [t.strip() for t in topics_txt.splitlines() if t.strip()],
        "questions": [q.strip() for q in questions_txt.splitlines() if q.strip()],
        "decisions": final_dec,
        "actions": final_act,
    }

    with_deadline = 0
    if len(final_act):
        with_deadline = int((~final_act["Deadline"].astype(str).str.lower().isin(["not specified", ""])).sum())
    st.markdown(
        f"""
<div class="metric-row">
  <div class="metric"><div class="num">{len(snapshot['attendees'])}</div><div class="lbl">Attendees</div></div>
  <div class="metric"><div class="num">{len(final_dec)}</div><div class="lbl">Decisions</div></div>
  <div class="metric"><div class="num">{len(final_act)}</div><div class="lbl">Action items</div></div>
  <div class="metric"><div class="num">{with_deadline}</div><div class="lbl">With a deadline</div></div>
</div>
""",
        unsafe_allow_html=True,
    )

    with tab_exp:
        st.markdown("Download your edited minutes in the format you need.")
        base = slug(snapshot["title"])
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            try:
                docx_bytes = build_docx(snapshot)
            except Exception:
                docx_bytes = None
            if docx_bytes:
                st.download_button(
                    "Word (.docx)",
                    data=docx_bytes,
                    file_name=f"{base}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True,
                )
            else:
                st.caption("Word export unavailable. Install python-docx.")
        with c2:
            st.download_button(
                "Markdown (.md)",
                data=build_markdown(snapshot).encode("utf-8"),
                file_name=f"{base}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with c3:
            st.download_button(
                "Actions (.csv)",
                data=final_act.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{base}_actions.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with c4:
            payload = {
                **{k: v for k, v in snapshot.items() if k not in ("decisions", "actions")},
                "decisions": final_dec.to_dict(orient="records"),
                "action_items": final_act.to_dict(orient="records"),
            }
            st.download_button(
                "JSON (.json)",
                data=json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8"),
                file_name=f"{base}.json",
                mime="application/json",
                use_container_width=True,
            )
        with st.expander("Preview Markdown"):
            st.markdown(build_markdown(snapshot))
