"""
Pip the Writing Coach  -  an AI writing coach built with Streamlit + Groq.

Users paste a paragraph. The app:
  1. corrects grammar, spelling and punctuation
  2. explains every mistake in friendly, simple language
  3. rewrites the paragraph in Simple English and/or Formal English

Run:
    pip install -r requirements.txt
    streamlit run app.py

API key (never hard-code it!). Use ONE of these:
    * paste it into the sidebar box inside the app
    * set an environment variable:  GROQ_API_KEY=your_key
    * create .streamlit/secrets.toml containing:  GROQ_API_KEY = "your_key"
"""

from __future__ import annotations

import difflib
import html
import json
import os
import re
from typing import Any

import streamlit as st

# --------------------------------------------------------------------------
# Page config must be the very first Streamlit call
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Pip the Writing Coach",
    page_icon="✏️",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    import groq
except ImportError:  # pragma: no cover
    st.error("The `groq` package is missing. Run: `pip install -r requirements.txt`")
    st.stop()

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
# Tried in order. If one model is unavailable / rate limited, the next is used.
DEFAULT_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
MAX_CHARS = 3000
STYLE_OPTIONS = {
    "Simple English": ["simple"],
    "Formal English": ["formal"],
    "Both": ["simple", "formal"],
}
STYLE_TITLES = {
    "simple": "🌱 Simple English",
    "formal": "🎓 Formal English",
}
EXAMPLE_TEXT = (
    "Yesterday me and my friend goes to the library for study. "
    "We was very tired but we still finish all of the homeworks. "
    "The teacher say that we should reads more book, because it make our "
    "vocabulary more better."
)

SYSTEM_PROMPT = """You are Pip, a warm, patient and encouraging English writing coach.
The learner gives you a paragraph. You must analyse it and reply with ONE valid JSON object and nothing else.

JSON schema:
{
  "corrected_text": "the full paragraph with all grammar, spelling, punctuation and capitalisation fixed. Keep the author's meaning and voice. Change as little as possible.",
  "score": 0-100 integer rating the ORIGINAL writing accuracy (100 = no mistakes),
  "summary": "one short, friendly, encouraging sentence about the writing",
  "mistakes": [
    {
      "original": "the exact wrong words copied from the learner's text (a short phrase, not a whole paragraph)",
      "correction": "the corrected words",
      "category": "one of: Grammar, Spelling, Punctuation, Capitalization, Tense, Articles, Word choice, Sentence structure",
      "explanation": "1-2 short, kind sentences explaining the rule in very simple words"
    }
  ],
  "rewrites": {
__REWRITE_KEYS__
  },
  "tips": ["2-3 short, practical tips for improving this learner's writing"]
}

Rules:
- Return raw JSON only. No markdown, no code fences, no extra text.
- List at most 12 mistakes, most important first. If the text is already perfect, use an empty list and a high score.
- Rewrites must be based on the corrected text and keep the same meaning.
- "simple" rewrite: short sentences, everyday words, easy for a beginner (CEFR A2-B1).
- "formal" rewrite: professional, polished, academic tone, no contractions.
- If the text is not written in English, still do your best: translate it into correct English for "corrected_text" and mention this in "summary".
- Never invent facts that are not in the learner's text.
"""


class CoachError(Exception):
    """Friendly, user-facing error."""


# --------------------------------------------------------------------------
# Styling (cute + study friendly, high contrast, big readable fonts)
# --------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@500;600;700&family=Nunito:wght@500;600;700;800&display=swap');

:root{
  --ink:#2B2A4C;
  --ink-soft:#55537A;
  --paper:#F6F3FF;
  --line:#E4DDFB;
  --lav:#CFC4FF;
  --purple:#6C5CE7;
  --purple-dark:#5446C9;
  --mint:#C9F2DF;
  --mint-ink:#0B5B3B;
  --peach:#FFE0D1;
  --butter:#FFF3B0;
  --rose:#FFD6DD;
  --rose-ink:#8A1C3A;
}

html, body, .stApp{
  font-family:'Nunito','Trebuchet MS',Verdana,sans-serif !important;
  color:var(--ink);
}
.stApp{
  background-color:var(--paper);
  background-image:linear-gradient(var(--line) 1px, transparent 1px);
  background-size:100% 34px;
}
.stApp p, .stApp li, .stApp label, .stApp span, .stApp div{ color:var(--ink); }
.stApp h1,.stApp h2,.stApp h3,.stApp h4{
  font-family:'Fredoka','Nunito',sans-serif !important;
  color:var(--ink) !important;
  letter-spacing:.2px;
}
.block-container{ padding-top:2rem; padding-bottom:3rem; max-width:1100px; }
#MainMenu, footer{ visibility:hidden; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"]{
  background:#EFE9FF !important;
  border-right:3px dashed var(--lav);
}
[data-testid="stSidebar"] *{ font-size:1.02rem; }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3{ font-size:1.5rem; }

/* ---------- Hero ---------- */
.hero{
  display:flex; align-items:center; gap:1.2rem;
  background:#FFFFFF;
  border:3px dashed var(--lav);
  border-radius:28px;
  padding:1.3rem 1.6rem;
  margin-bottom:1.4rem;
  box-shadow:6px 6px 0 var(--lav);
}
.hero .mascot{
  flex:0 0 auto; width:76px; height:76px; border-radius:50%;
  background:var(--butter); border:3px solid var(--ink);
  display:flex; align-items:center; justify-content:center;
  font-size:2.4rem; transform:rotate(-8deg);
}
.hero h1{ margin:0; font-size:2.4rem; line-height:1.1; }
.hero p{ margin:.35rem 0 0 0; font-size:1.15rem; color:var(--ink-soft) !important; font-weight:600; }

/* ---------- Section titles ---------- */
.step-title{
  font-family:'Fredoka',sans-serif; font-weight:600; font-size:1.45rem;
  margin:.4rem 0 .6rem 0; color:var(--ink);
}

/* ---------- Text area (sticky note feel) ---------- */
.stTextArea textarea{
  font-family:'Nunito',sans-serif !important;
  font-size:1.2rem !important; line-height:1.7 !important; font-weight:600 !important;
  color:var(--ink) !important; background:#FFFDF2 !important;
  border:3px solid var(--ink) !important; border-radius:20px !important;
  padding:1rem 1.1rem !important;
  box-shadow:5px 5px 0 var(--butter);
}
.stTextArea textarea::placeholder{ color:#7C79A3 !important; opacity:1; }
.stTextArea textarea:focus{ border-color:var(--purple) !important; box-shadow:5px 5px 0 var(--lav) !important; }
.stTextArea [data-baseweb="textarea"], .stTextArea [data-baseweb="base-input"]{ background:transparent !important; }

/* ---------- Radio ---------- */
[data-testid="stRadio"] label p{ font-size:1.1rem !important; font-weight:700 !important; color:var(--ink) !important; }

/* ---------- Inputs in sidebar ---------- */
[data-testid="stSidebar"] input{
  font-size:1rem !important; color:var(--ink) !important; background:#FFFFFF !important;
}

/* ---------- Buttons ---------- */
.stButton button, .stDownloadButton button{
  font-family:'Fredoka',sans-serif !important; font-size:1.15rem !important; font-weight:600 !important;
  border-radius:999px !important; border:3px solid var(--ink) !important;
  padding:.55rem 1.4rem !important; background:#FFFFFF !important; color:var(--ink) !important;
  box-shadow:4px 4px 0 var(--ink); transition:transform .08s ease, box-shadow .08s ease;
}
.stButton button p, .stDownloadButton button p{ color:inherit !important; font-size:1.15rem !important; font-family:'Fredoka',sans-serif !important; }
.stButton button:hover, .stDownloadButton button:hover{
  transform:translate(-1px,-1px); box-shadow:6px 6px 0 var(--ink); border-color:var(--ink) !important;
}
.stButton button:active, .stDownloadButton button:active{ transform:translate(3px,3px); box-shadow:1px 1px 0 var(--ink); }
.stButton button[kind="primary"]{ background:var(--purple) !important; color:#FFFFFF !important; }
.stButton button[kind="primary"] p{ color:#FFFFFF !important; }
.stButton button[kind="primary"]:hover{ background:var(--purple-dark) !important; }
.stButton button:focus-visible, .stDownloadButton button:focus-visible{ outline:4px solid var(--butter) !important; outline-offset:2px; }

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"]{ gap:.5rem; border-bottom:3px dashed var(--lav); padding-bottom:.4rem; }
.stTabs [data-baseweb="tab"]{
  font-family:'Fredoka',sans-serif !important; font-size:1.2rem !important; font-weight:600 !important;
  background:#FFFFFF; border:3px solid var(--ink); border-radius:999px; padding:.35rem 1.2rem; height:auto;
}
.stTabs [data-baseweb="tab"] p{ font-size:1.2rem !important; font-family:'Fredoka',sans-serif !important; color:var(--ink) !important; }
.stTabs [aria-selected="true"]{ background:var(--butter) !important; box-shadow:3px 3px 0 var(--ink); }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"]{ display:none; }

/* ---------- Result cards ---------- */
.card{
  background:#FFFFFF; border:3px solid var(--ink); border-radius:22px;
  padding:1.1rem 1.3rem; margin:.7rem 0; font-size:1.18rem; line-height:1.85; font-weight:600;
  box-shadow:5px 5px 0 var(--lav);
}
.card.mint{ background:#F1FCF7; box-shadow:5px 5px 0 var(--mint); }
.card.peach{ background:#FFF7F2; box-shadow:5px 5px 0 var(--peach); }
.label{ font-family:'Fredoka',sans-serif; font-weight:600; font-size:1.1rem; color:var(--ink-soft); margin-bottom:.2rem; }
.del{ background:var(--rose); color:var(--rose-ink); text-decoration:line-through; border-radius:8px; padding:0 .3rem; font-weight:700; }
.ins{ background:var(--mint); color:var(--mint-ink); border-radius:8px; padding:0 .3rem; font-weight:800; }

.mistake{
  background:#FFFFFF; border:3px solid var(--ink); border-radius:22px;
  padding:1rem 1.2rem; margin:.8rem 0; box-shadow:5px 5px 0 var(--rose);
}
.mistake .chip{
  display:inline-block; font-family:'Fredoka',sans-serif; font-weight:600; font-size:.98rem;
  background:var(--lav); color:var(--ink); border:2px solid var(--ink); border-radius:999px; padding:.1rem .8rem;
}
.mistake .fix{ font-size:1.3rem; margin:.6rem 0 .3rem 0; font-weight:700; line-height:1.9; }
.mistake .why{ font-size:1.12rem; line-height:1.7; margin:.2rem 0 0 0; font-weight:600; color:var(--ink) !important; }

.score{
  display:flex; align-items:center; gap:1.1rem; background:#FFFFFF;
  border:3px solid var(--ink); border-radius:24px; padding:1rem 1.3rem; margin:.5rem 0 1rem 0;
  box-shadow:5px 5px 0 var(--butter);
}
.score .ring{
  flex:0 0 auto; width:84px; height:84px; border-radius:50%; border:4px solid var(--ink);
  display:flex; align-items:center; justify-content:center;
  font-family:'Fredoka',sans-serif; font-weight:700; font-size:1.9rem; color:var(--ink);
}
.score .big{ font-family:'Fredoka',sans-serif; font-weight:600; font-size:1.45rem; }
.score .sub{ font-size:1.12rem; font-weight:600; color:var(--ink-soft) !important; }

.tips{ background:var(--butter); border:3px solid var(--ink); border-radius:22px; padding:1rem 1.3rem; margin-top:1rem; }
.tips h4{ margin:0 0 .4rem 0; font-size:1.3rem; }
.tips ul{ margin:0; padding-left:1.3rem; }
.tips li{ font-size:1.12rem; font-weight:700; line-height:1.7; }

.celebrate{
  background:var(--mint); border:3px solid var(--ink); border-radius:22px; padding:1rem 1.3rem;
  font-size:1.2rem; font-weight:700; color:var(--mint-ink) !important;
}
.small-note{ font-size:.98rem; color:var(--ink-soft) !important; font-weight:600; }
.counter{ text-align:right; font-size:1rem; font-weight:700; color:var(--ink-soft) !important; margin-top:-.4rem; }

/* alerts keep readable text */
[data-testid="stAlert"] p{ font-size:1.08rem !important; font-weight:700 !important; }

@media (max-width: 640px){
  .hero{ flex-direction:column; text-align:center; }
  .hero h1{ font-size:1.9rem; }
}
@media (prefers-reduced-motion: reduce){
  .stButton button, .stDownloadButton button{ transition:none; }
}
</style>
"""


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def esc(value: Any) -> str:
    """HTML-escape any value and keep line breaks."""
    return html.escape(str(value if value is not None else "")).replace("\n", "<br>")


def get_api_key(typed_key: str) -> str:
    """Priority: sidebar input > Streamlit secrets > environment variable."""
    if typed_key and typed_key.strip():
        return typed_key.strip()
    try:
        secret = st.secrets.get("GROQ_API_KEY", "")  # may raise if no secrets file
        if secret:
            return str(secret).strip()
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY", "").strip()


def get_models() -> list[str]:
    custom = os.environ.get("GROQ_MODEL", "").strip()
    models = [custom] if custom else []
    models += [m for m in DEFAULT_MODELS if m != custom]
    return models


def build_system_prompt(styles: list[str]) -> str:
    descriptions = {
        "simple": '    "simple": "the corrected paragraph rewritten in Simple English"',
        "formal": '    "formal": "the corrected paragraph rewritten in Formal English"',
    }
    keys = ",\n".join(descriptions[s] for s in styles)
    return SYSTEM_PROMPT.replace("__REWRITE_KEYS__", keys)


def call_groq(api_key: str, system_prompt: str, user_text: str) -> str:
    """Call Groq with automatic model fallback. Raises CoachError with a friendly message."""
    client = groq.Groq(api_key=api_key, timeout=60.0, max_retries=2)
    last_problem = "Pip could not get an answer from the AI service."

    for model in get_models():
        for json_mode in (True, False):
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Here is the learner's paragraph:\n\n{user_text}"},
                ],
                "temperature": 0.3,
                "max_tokens": 3000,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            try:
                response = client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content if response.choices else ""
                if content and content.strip():
                    return content
                last_problem = "The AI returned an empty answer."
            except (groq.AuthenticationError, groq.PermissionDeniedError):
                raise CoachError(
                    "Your Groq API key was rejected. Please check it in the sidebar "
                    "(create a key at console.groq.com/keys)."
                )
            except groq.RateLimitError:
                last_problem = "Groq is busy or the rate limit was reached. Please wait a moment and try again."
                break  # limits are per model -> try the next model
            except (groq.BadRequestError, groq.NotFoundError, groq.UnprocessableEntityError):
                last_problem = "The AI model could not handle this request."
                continue  # retry without JSON mode, then next model
            except groq.APIConnectionError:
                raise CoachError(
                    "Could not reach Groq. Check your internet connection and try again."
                )
            except groq.APIStatusError:
                last_problem = "The AI service had a temporary problem. Please try again."
                break
            except Exception:
                last_problem = "Something unexpected happened while talking to the AI."
                break
    raise CoachError(last_problem)


def extract_json(raw: str) -> dict:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
    except ValueError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("No JSON object found")
        data = json.loads(cleaned[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("JSON is not an object")
    return data


def normalize(data: dict, original: str, styles: list[str]) -> dict:
    """Make sure the result always has the expected shape, whatever the model returned."""
    corrected = str(data.get("corrected_text") or "").strip() or original

    try:
        score = int(round(float(data.get("score", 0))))
    except (TypeError, ValueError):
        score = 0
    score = max(0, min(100, score))

    mistakes = []
    raw_mistakes = data.get("mistakes")
    if isinstance(raw_mistakes, list):
        for item in raw_mistakes[:12]:
            if not isinstance(item, dict):
                continue
            wrong = str(item.get("original") or "").strip()
            right = str(item.get("correction") or "").strip()
            if not wrong and not right:
                continue
            mistakes.append(
                {
                    "original": wrong,
                    "correction": right,
                    "category": str(item.get("category") or "Grammar").strip() or "Grammar",
                    "explanation": str(item.get("explanation") or "").strip()
                    or "This change makes the sentence correct and clear.",
                }
            )

    raw_rewrites = data.get("rewrites") if isinstance(data.get("rewrites"), dict) else {}
    rewrites = {}
    for style in styles:
        text = raw_rewrites.get(style)
        text = str(text).strip() if text else ""
        rewrites[style] = text or corrected

    tips = []
    if isinstance(data.get("tips"), list):
        tips = [str(t).strip() for t in data["tips"] if str(t).strip()][:3]

    return {
        "original": original,
        "corrected": corrected,
        "score": score,
        "summary": str(data.get("summary") or "").strip(),
        "mistakes": mistakes,
        "rewrites": rewrites,
        "tips": tips,
        "styles": styles,
    }


def coach(api_key: str, text: str, styles: list[str]) -> dict:
    system_prompt = build_system_prompt(styles)
    last_error: Exception | None = None
    for _ in range(2):  # one retry if the model answers with broken JSON
        raw = call_groq(api_key, system_prompt, text)
        try:
            return normalize(extract_json(raw), text, styles)
        except (ValueError, TypeError, KeyError) as exc:
            last_error = exc
    raise CoachError(
        "Pip got a messy answer from the AI and could not read it. Please press the button again."
    ) from last_error


def diff_html(original: str, corrected: str) -> tuple[str, str]:
    """Word-level diff. Returns (original_html, corrected_html) with highlights."""
    a, b = original.split(), corrected.split()
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    left, right = [], []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            chunk = html.escape(" ".join(a[i1:i2]))
            left.append(chunk)
            right.append(chunk)
        else:
            if i2 > i1:
                left.append(f'<span class="del">{html.escape(" ".join(a[i1:i2]))}</span>')
            if j2 > j1:
                right.append(f'<span class="ins">{html.escape(" ".join(b[j1:j2]))}</span>')
    return " ".join(left), " ".join(right)


def score_info(score: int) -> tuple[str, str, str]:
    if score >= 90:
        return "Superstar writing! 🌟", "Almost perfect. Just tiny polish left.", "#C9F2DF"
    if score >= 75:
        return "Great job! 🎉", "Solid writing with a few things to fix.", "#E3F7C8"
    if score >= 50:
        return "Getting there! 💪", "Good ideas. Let's tidy up the grammar.", "#FFF3B0"
    return "Keep practising! 🌱", "Every mistake is a chance to learn.", "#FFE0D1"


def build_report(result: dict) -> str:
    lines = [
        "PIP THE WRITING COACH - REPORT",
        "=" * 34,
        f"Score: {result['score']}/100",
        "",
        "ORIGINAL",
        result["original"],
        "",
        "CORRECTED",
        result["corrected"],
        "",
        "MISTAKES EXPLAINED",
    ]
    if result["mistakes"]:
        for i, m in enumerate(result["mistakes"], 1):
            lines.append(f"{i}. [{m['category']}] {m['original']} -> {m['correction']}")
            lines.append(f"   {m['explanation']}")
    else:
        lines.append("No mistakes found!")
    for style in result["styles"]:
        lines += ["", STYLE_TITLES[style].split(" ", 1)[1].upper(), result["rewrites"][style]]
    if result["tips"]:
        lines += ["", "TIPS"] + [f"- {t}" for t in result["tips"]]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Session state + callbacks
# --------------------------------------------------------------------------
st.session_state.setdefault("text_input", "")
st.session_state.setdefault("result", None)
st.session_state.setdefault("coached_count", 0)


def load_example() -> None:
    st.session_state["text_input"] = EXAMPLE_TEXT
    st.session_state["result"] = None


def clear_all() -> None:
    st.session_state["text_input"] = ""
    st.session_state["result"] = None


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.markdown(CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🎒 Pip's desk")
    typed_key = st.text_input(
        "Groq API key",
        type="password",
        placeholder="gsk_...",
        help="Your key stays in this browser session only. Get one free at console.groq.com/keys",
    )
    api_key = get_api_key(typed_key)
    if api_key:
        st.success("Key ready ✔️")
    else:
        st.warning("Add your Groq API key to start coaching.")

    st.markdown("---")
    st.markdown("### 📚 How it works")
    st.markdown(
        "1. Paste your paragraph\n"
        "2. Pick a rewrite style\n"
        "3. Press **Coach my writing**\n"
        "4. Read the fixes and learn why"
    )
    st.markdown("---")
    st.markdown(f"**📝 Paragraphs coached:** {st.session_state['coached_count']}")
    st.markdown(
        '<p class="small-note">Tip: short paragraphs (3-6 sentences) give the clearest lessons.</p>',
        unsafe_allow_html=True,
    )

st.markdown(
    """
<div class="hero">
  <div class="mascot">✏️</div>
  <div>
    <h1>Pip the Writing Coach</h1>
    <p>Paste a paragraph. Pip fixes it, explains each mistake, and rewrites it your way.</p>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown('<div class="step-title">📝 Your paragraph</div>', unsafe_allow_html=True)
    st.text_area(
        "Your paragraph",
        key="text_input",
        height=240,
        max_chars=MAX_CHARS,
        placeholder="Type or paste your paragraph here...",
        label_visibility="collapsed",
    )
    chars = len(st.session_state["text_input"])
    st.markdown(f'<div class="counter">{chars} / {MAX_CHARS} characters</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="step-title">🎨 Rewrite style</div>', unsafe_allow_html=True)
    style_choice = st.radio(
        "Rewrite style",
        list(STYLE_OPTIONS.keys()),
        index=2,
        label_visibility="collapsed",
    )
    st.markdown(
        '<p class="small-note">Simple English = short and easy words.<br>'
        "Formal English = polished and professional.</p>",
        unsafe_allow_html=True,
    )

b1, b2, b3, _ = st.columns([2.2, 1.4, 1.2, 2.2])
with b1:
    analyze = st.button("✨ Coach my writing", type="primary", use_container_width=True)
with b2:
    st.button("📖 Try example", on_click=load_example, use_container_width=True)
with b3:
    st.button("🧽 Clear", on_click=clear_all, use_container_width=True)

# ---- Run the coach ----
if analyze:
    text = st.session_state["text_input"].strip()
    if not api_key:
        st.error("Please add your Groq API key in the sidebar first. 🔑")
    elif not text:
        st.warning("Write or paste a paragraph first, then press the button. ✍️")
    elif len(text.split()) < 2:
        st.warning("Please write at least a short sentence so Pip has something to coach.")
    elif len(text) > MAX_CHARS:
        st.warning(f"That's a bit long. Please keep it under {MAX_CHARS} characters.")
    else:
        try:
            with st.spinner("Pip is reading your paragraph... 📖"):
                st.session_state["result"] = coach(api_key, text, STYLE_OPTIONS[style_choice])
            st.session_state["coached_count"] += 1
        except CoachError as exc:
            st.session_state["result"] = None
            st.error(str(exc))
        except Exception:
            st.session_state["result"] = None
            st.error("Something unexpected happened. Please try again in a moment.")

# ---- Show results ----
result = st.session_state.get("result")
if result:
    st.markdown("---")
    title, sub, color = score_info(result["score"])
    st.markdown(
        f"""
<div class="score">
  <div class="ring" style="background:{color}">{result['score']}</div>
  <div>
    <div class="big">{esc(title)}</div>
    <div class="sub">{esc(result['summary'] or sub)}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    tab_fix, tab_why, tab_rewrite = st.tabs(["✅ Corrected", "🔍 Mistakes explained", "✨ Rewrites"])

    with tab_fix:
        before_html, after_html = diff_html(result["original"], result["corrected"])
        st.markdown(
            f'<div class="label">Your original (mistakes in pink)</div><div class="card peach">{before_html}</div>'
            f'<div class="label">Pip\'s corrected version (fixes in green)</div><div class="card mint">{after_html}</div>',
            unsafe_allow_html=True,
        )

    with tab_why:
        if not result["mistakes"]:
            st.markdown(
                '<div class="celebrate">🎉 No mistakes found. Wonderful writing!</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<p class="small-note">Pip found {len(result["mistakes"])} thing(s) to learn from:</p>',
                unsafe_allow_html=True,
            )
            for m in result["mistakes"]:
                st.markdown(
                    f'<div class="mistake"><span class="chip">{esc(m["category"])}</span>'
                    f'<div class="fix"><span class="del">{esc(m["original"] or "(missing)")}</span> '
                    f'&nbsp;➜&nbsp; <span class="ins">{esc(m["correction"] or "(remove)")}</span></div>'
                    f'<p class="why">💡 {esc(m["explanation"])}</p></div>',
                    unsafe_allow_html=True,
                )

    with tab_rewrite:
        for style in result["styles"]:
            st.markdown(
                f'<div class="label">{STYLE_TITLES[style]}</div>'
                f'<div class="card">{esc(result["rewrites"][style])}</div>',
                unsafe_allow_html=True,
            )

    if result["tips"]:
        items = "".join(f"<li>{esc(t)}</li>" for t in result["tips"])
        st.markdown(
            f'<div class="tips"><h4>🌟 Pip\'s study tips</h4><ul>{items}</ul></div>',
            unsafe_allow_html=True,
        )

    st.download_button(
        "💾 Download my report",
        data=build_report(result),
        file_name="writing_coach_report.txt",
        mime="text/plain",
    )
else:
    st.markdown(
        '<p class="small-note">Your results will appear here. Try the example if you want to see how it works. 🌈</p>',
        unsafe_allow_html=True,
    )
