"""Kuwentuhan Bot: full system. Chat (streaming), language modes, logging, ratings, admin, rate limit."""
import os, json, time, csv, io, sqlite3, secrets, collections, asyncio, random
from typing import Literal
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from anthropic import AsyncAnthropic

def load_env():
    if os.path.exists(".env"):
        with open(".env") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    os.environ[_k.strip()] = _v.strip().strip("'\"")

load_env()
PROVIDER = os.getenv("PROVIDER", "anthropic").lower()
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5") if PROVIDER != "openai" else os.getenv("OPENAI_MODEL", "gemma4:12b")
LOG_CHATS = os.getenv("LOG_CHATS", "true").lower() == "true"
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
RATE = int(os.getenv("RATE_LIMIT_PER_MIN", "20"))
DB_PATH = os.getenv("DB_PATH", "chats.db")

import re

BASE = (
    "You are the Taglish Tutor, a dedicated bilingual conversation coach whose only purpose is language learning and practice. "
    "You specialize exclusively in three interconnected language styles: standard English, Tagalog (Filipino), and everyday conversational Taglish.\n\n"
    "Your entire focus is divided between two types of learners:\n"
    "1. Filipino speakers who want to build confidence, polish their grammar, expand their vocabulary, and master natural-sounding English for work, school, or daily conversation.\n"
    "2. Foreigners and non-native speakers who want to learn real, practical Filipino, understand cultural phrases, and feel comfortable speaking Tagalog or modern Taglish.\n\n"
    "CURRICULUM & KNOWLEDGE BASE:\n"
    "• Parts of Speech & Contrast: English is Subject-Verb-Object ('I ate lunch'), Tagalog is usually Verb-first ('Kumain ako ng tanghalian'). Tagalog pronouns like 'Siya' are gender-neutral (both he and she). Tagalog adjectives often use 'Ma-' (maganda, masarap). 'Sa' is the universal preposition for in/on/at/to.\n"
    "• The 3 Types of 'No': 'Hindi' (facts/negation: Hindi ako pagod), 'Ayaw' (personal dislike/refusal: Ayaw ko niyan), 'Huwag' (commands/prohibition: Huwag kang tumakbo).\n"
    "• Untranslatable particles: na (already/now), pa (still/yet), naman (softener/turn-taking), pala (realization/surprise), daw/raw (hearsay/they said), ba (question marker).\n"
    "• Politeness: Po, opo, honorifics (ate, kuya).\n"
    "• Common Filipinisms to polish: 'turn on the light' (not 'open the light'), 'turn off the computer' (not 'close the computer'), 'I am heading out / leaving now' (not 'I will go ahead'), 'fill out the form' (not 'fill up').\n"
    "• Real-world scenarios: Dining & ordering food (Almusal, Tanghalian, Hapunan, 'Pahingi po ng...', 'Pa-bill out po', 'Pa-balot po'), Commuting ('Bayad po', 'Sukli po', 'Para po / Sa tabi lang po', 'Dadaan po ba sa...').\n\n"
    "Your behavior during every interaction must strictly follow these rules:\n\n"
    "First, maintain an active, engaging conversation. Whenever the user speaks, answer their message naturally to keep the dialogue flowing like a real conversation between friends. "
    "Ask follow-up questions or give short practice prompts to give them more opportunities to speak. If they ask to learn basics or select a beginner topic, teach it clearly with simple examples and invite them to try a practice sentence.\n\n"
    "Second, provide gentle, supportive feedback whenever there is a grammar mistake, awkward word choice, or unnatural phrasing. "
    "Never make the learner feel embarrassed or discouraged. Point out the error kindly, provide the natural correction, and give a quick, simple explanation of why. "
    "Offer both formal options and casual, everyday Taglish alternatives so the learner understands when to use each "
    "(formatted warmly at the end of your response, e.g. using '💡 Quick Tip:' for English or '✨ Filipino Tip:' for Tagalog).\n\n"
    "PRESENTATION & UI FORMATTING RULES:\n"
    "• Whenever explaining comparisons, differences, starter kits, vocabulary sets, or grammar rules, ALWAYS format them as Markdown Tables (e.g. '| Category | Example | What It Means |' or '| Level | English | Tagalog / Taglish |'). The UI renders these tables into beautiful boxed cards with clean borders and columns matching the visual design.\n"
    "• Use '## ' for main section titles (e.g. '## 1️⃣ Tagalog Basics') and '### ' for practice prompts or sub-topics.\n"
    "• Use '---' horizontal rules to divide lesson parts cleanly.\n"
    "• Format tips, quizzes, and practice challenges at the end with '💡 Quick Tip:', '✨ Filipino Tip:', or '### Practice Prompt'.\n\n"
    "Always maintain a warm, welcoming, patient, and encouraging tone. Celebrate small improvements with positive words like \"Galing!\", \"Nice try!\", and \"You're getting better!\" to keep the learner motivated and comfortable making mistakes."
)
PROMPTS = {
    "fil": BASE + "\n\nCURRENT MODE: The learner is practicing Filipino (Tagalog). Reply in learner-friendly Tagalog, and gently coach Tagalog grammar, affixes, and natural expressions.",
    "en": BASE + "\n\nCURRENT MODE: The learner is practicing English. Reply in natural, encouraging English, and gently point out any English grammar, vocabulary, or phrasing improvements.",
    "both": BASE + "\n\nCURRENT MODE: The learner is practicing bilingual Taglish. Reply in natural Taglish, coaching them on both English and Filipino fluency and natural code-switching.",
    "auto": BASE + "\n\nCURRENT MODE: Automatically detect whether the learner is practicing English, Filipino, or Taglish, adapt to their language, and gently coach their mistakes.",
}

# ---------- database ----------
def db():
    c = sqlite3.connect(DB_PATH); c.row_factory = sqlite3.Row; return c

with db() as c:
    c.execute("""CREATE TABLE IF NOT EXISTS chats(
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, session TEXT, mode TEXT, model TEXT,
        user_text TEXT, reply TEXT DEFAULT '', rating INTEGER DEFAULT 0)""")

# ---------- app ----------
class Msg(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)

class ChatRequest(BaseModel):
    mode: Literal["auto", "fil", "en", "both"] = "auto"
    session: str = Field(default="", max_length=64)
    messages: list[Msg] = Field(min_length=1, max_length=60)

class Rating(BaseModel):
    id: int
    value: Literal[-1, 1]

app = FastAPI(title="Kuwentuhan Bot")
app.mount("/static", StaticFiles(directory="static"), name="static")
hits: dict[str, collections.deque] = collections.defaultdict(collections.deque)
_client = None

def limited(request: Request) -> bool:
    ip = (request.headers.get("x-forwarded-for") or (request.client.host if request.client else "?")).split(",")[0].strip()
    q, now = hits[ip], time.time()
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE:
        return True
    q.append(now)
    return False

def admin_ok(request: Request) -> bool:
    tok = request.headers.get("x-admin-token") or request.query_params.get("token", "")
    return bool(ADMIN_TOKEN) and secrets.compare_digest(tok, ADMIN_TOKEN)

# ---------- providers ----------
def client() -> AsyncAnthropic:
    global _client
    if _client is None:
        _client = AsyncAnthropic()
    return _client

async def stream_anthropic(system, history):
    async with client().messages.stream(model=MODEL, max_tokens=900, system=system, messages=history) as s:
        async for text in s.text_stream:
            yield text

import urllib.parse

async def stream_openai(system, history):
    base = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1").rstrip("/")
    headers = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', 'ollama')}"}
    body = {"model": MODEL, "stream": True, "max_tokens": 900,
            "messages": [{"role": "system", "content": system}] + history}
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as http:
        yielded = False
        try:
            async with http.stream("POST", f"{base}/chat/completions", headers=headers, json=body) as r:
                if r.status_code == 200:
                    async for line in r.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[6:].strip()
                        if data == "[DONE]":
                            break
                        choices = json.loads(data).get("choices") or []
                        if choices and (d := choices[0].get("delta", {}).get("content")):
                            yielded = True
                            yield d
                elif "pollinations.ai" not in base:
                    raise RuntimeError(f"Provider returned {r.status_code}: {(await r.aread())[:200]!r}")
        except Exception:
            if "pollinations.ai" not in base:
                raise
        if not yielded and "pollinations.ai" in base:
            convo = "\n".join(f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}" for m in history[-6:])
            url = (
                f"https://text.pollinations.ai/{urllib.parse.quote(convo)}"
                f"?model={urllib.parse.quote(MODEL)}&system={urllib.parse.quote(system)}"
            )
            r2 = await http.get(url)
            if r2.status_code != 200 or not r2.text.strip():
                raise RuntimeError(f"Pollinations returned {r2.status_code}")
            words = r2.text.strip().split(" ")
            for i, w in enumerate(words):
                yield w + (" " if i < len(words) - 1 else "")
                await asyncio.sleep(0.02)

def _extract_name(text: str) -> str:
    m = re.search(r"\b(?:ako\s+(?:nga\s+pala\s+)?si|pangalan\s+ko\s+(?:po\s+)?ay|my\s+name\s+is|call\s+me|tawagin\s+mo\s+akong)\s+([a-zA-Z]{2,15})\b", text, re.I)
    if m and m.group(1).lower() not in {"sad", "tired", "happy", "hungry", "bored", "fine", "good", "okay", "ok", "here"}:
        return m.group(1).capitalize()
    return ""

def _extract_style(raw: str, history: list[dict]) -> dict:
    words = re.findall(r"[a-zA-ZÀ-ÿ']+", raw.lower())
    word_set = set(words)
    polite = bool(word_set & {"po", "opo", "ho", "oho"})
    slang_terms = ["lods", "lodibels", "pre", "pare", "tol", "bro", "sis", "bes", "besh", "beh", "mare", "idol", "baks"]
    slang = next((s for s in slang_terms if s in word_set), "")
    playful = bool(re.search(r"(haha|hehe|hihi|lol|lmao|😂|🤣|😆|😄|!{2,})", raw.lower()))
    name = ""
    for m in history:
        if m["role"] == "user" and (n := _extract_name(m["content"])):
            name = n
    return {"polite": polite, "slang": slang, "playful": playful, "name": name}

def _reflect_user_text(raw: str, mode: str) -> str:
    cleaned = re.sub(r"^(?:(?:uy|hey|hi|hello|uhm|hmm|ano|wait|btw|nga\s+pala|alam\s+mo\s+ba|alam\s+mo|bro|pre|pare|tol|lods|lodibels|sis|bes|besh|beh|mare|idol|baks|po|opo)[,\s!.]*)+", "", raw.strip(), flags=re.I)
    cleaned = re.sub(r"[\s,.!]*(?:haha+|hehe+|hihi+|lol+|lmao+|po|opo|ho|lods|pre|pare|bro|bes|beh)[\s.!?]*$", "", cleaned, flags=re.I)
    cleaned = cleaned.strip().rstrip(".!?")
    if not cleaned:
        return raw.strip().rstrip(".!?")
    if mode == "en":
        swaps = [
            (r"\bi\s+am\b", "you're"), (r"\bi'm\b", "you're"), (r"\bim\b", "you're"),
            (r"\bi\s+was\b", "you were"), (r"\bi\s+have\b", "you have"), (r"\bi've\b", "you've"),
            (r"\bi\s+feel\b", "you feel"), (r"\bi\s+like\b", "you like"), (r"\bi\s+love\b", "you love"),
            (r"\bi\s+want\b", "you want"), (r"\bi\s+need\b", "you need"), (r"\bi\s+just\b", "you just"),
            (r"\bmy\b", "your"), (r"\bme\b", "you"), (r"\bmyself\b", "yourself"), (r"\bi\b", "you"),
        ]
    else:
        swaps = [
            (r"\bpo\s+ako\b", "ka"), (r"\bpo\s+kami\b", "kayo"), (r"\bpo\s+ko\b", "mo"),
            (r"\bako'y\b", "ikaw ay"), (r"\bako\s+ay\b", "ikaw ay"),
            (r"\bako\s+gusto\b", "gusto mo"),
            (r"\bgusto\s+ko\b", "gusto mo"), (r"\bayaw\s+ko\b", "ayaw mo"),
            (r"\bmahilig\s+ako\b", "mahilig ka"), (r"\bkumakain\s+ako\b", "kumakain ka"),
            (r"\bkumain\s+ako\b", "kumain ka"), (r"\bnag-aaral\s+ako\b", "nag-aaral ka"),
            (r"\bnaglalaro\s+ako\b", "naglalaro ka"), (r"\bnanonood\s+ako\b", "nanonood ka"),
            (r"\bnandito\s+ako\b", "nandiyan ka"), (r"\bpupunta\s+ako\b", "pupunta ka"),
            (r"\bgaling\s+ako\b", "galing ka"), (r"\bpagod\s+ako\b", "pagod ka"),
            (r"\bgutom\s+ako\b", "gutom ka"), (r"\bmasaya\s+ako\b", "masaya ka"),
            (r"\bko\b", "mo"), (r"\bakin\b", "iyo"), (r"\baking\b", "iyong"),
            (r"\bnamin\b", "ninyo"), (r"\bkami\b", "kayo"), (r"\bako\b", "ka"),
            (r"\bi\s+am\b", "you're"), (r"\bi'm\b", "you're"), (r"\bmy\b", "your"), (r"\bi\b", "you"),
        ]
    out = cleaned
    for pat, rep in swaps:
        out = re.sub(pat, rep, out, flags=re.I)
    if out and out[0].isupper() and not out.startswith("I "):
        out = out[0].lower() + out[1:]
    return out

def _check_language_shift(text: str) -> str:
    low = text.lower().strip()
    clean_low = re.sub(r"[^\w\s-]", "", low).strip()

    # Exact short replies (e.g. answering "What language would you like to practice today?")
    if clean_low in (
        "tagalog", "filipino", "tagalog po", "filipino po", "tagalog please", "filipino please",
        "tagalog lang", "filipino lang", "tagalog muna", "filipino muna", "tagalog na lang",
        "tagalog tayo", "filipino tayo", "magtagalog", "mag-tagalog", "mag tagalog"
    ):
        return "fil"
    if clean_low in (
        "english", "english po", "english please", "english lang", "english muna",
        "english na lang", "english tayo", "magenglish", "mag-english", "mag english"
    ):
        return "en"
    if clean_low in (
        "taglish", "both", "taglish po", "both po", "taglish please", "both please",
        "taglish lang", "both lang", "taglish muna", "taglish na lang", "taglish tayo",
        "magtaglish", "mag-taglish", "mag taglish"
    ):
        return "both"

    # Taglish / Both shift patterns
    taglish_patterns = [
        r"\b(?:taglish|both|mix)\b",
        r"\b(?:mag|mag-|mag\s+)taglish\b",
        r"\b(?:both|taglish)\s+(?:please|naman|tayo|muna|mode|lang|na\s+lang)\b",
        r"\b(?:english\s+and\s+tagalog|tagalog\s+and\s+english|filipino\s+and\s+english|english\s+and\s+filipino)\b",
        r"\b(?:both\s+languages|both\s+english\s+and\s+tagalog)\b",
    ]
    if any(re.search(p, low) for p in taglish_patterns):
        return "both"

    # Tagalog/Filipino shift patterns
    tagalog_patterns = [
        r"\b(?:learn|study|practice|teach|speak|talk|switch|change|shift|do)\s+(?:in\s+|to\s+|me\s+)?(?:tagalog|filipino)\b",
        r"\b(?:tagalog|filipino)\s+(?:please|naman|tayo|mode|muna|lang|na\s+lang)\b",
        r"\b(?:mag|mag-|mag\s+)(?:tagalog|filipino)\b",
        r"\b(?:gusto\s+ko|gustong|gusto\s+kong)\s+(?:matuto|mag-aral|mag-practice|magpaturo|magsalita)\s+(?:ng\s+|mag-?)?(?:tagalog|filipino)\b",
        r"\b(?:gusto\s+ko|gustong|gusto\s+kong)\s+mag-?(?:tagalog|filipino)\b",
        r"\bturuan\s+mo\s+(?:ako|ko)\s+(?:ng\s+|mag-?)?(?:tagalog|filipino)\b",
        r"\b(?:pwede\s+ba|can\s+we|let\s*s|lets)\s+(?:mag-?)?(?:tagalog|filipino)\b",
        r"\b(?:i\s+want\s+to|i\s+wanna|id\s+like\s+to|want\s+to|wanna)\s+(?:practice|learn|study|speak|talk\s+in)\s+(?:in\s+)?(?:tagalog|filipino)\b",
        r"\b(?:help\s+me\s+with|help\s+me\s+learn|help\s+me\s+speak)\s+(?:tagalog|filipino)\b",
        r"\b(?:how\s+to\s+speak|teach\s+me\s+how\s+to\s+speak)\s+(?:tagalog|filipino)\b",
        r"\bmag-?usap\s+tayo\s+sa\s+(?:tagalog|filipino)\b",
    ]
    if any(re.search(p, low) for p in tagalog_patterns):
        return "fil"

    # English shift patterns
    english_patterns = [
        r"\b(?:learn|study|practice|teach|speak|talk|switch|change|shift|do)\s+(?:in\s+|to\s+|me\s+)?english\b",
        r"\benglish\s+(?:please|naman|tayo|mode|muna|lang|na\s+lang)\b",
        r"\b(?:mag|mag-|mag\s+)english\b",
        r"\b(?:gusto\s+ko|gustong|gusto\s+kong)\s+(?:matuto|mag-aral|mag-practice|magpaturo|magsalita)\s+(?:ng\s+|mag-?)?english\b",
        r"\b(?:gusto\s+ko|gustong|gusto\s+kong)\s+mag-?english\b",
        r"\bturuan\s+mo\s+(?:ako|ko)\s+(?:ng\s+|mag-?)?english\b",
        r"\b(?:pwede\s+ba|can\s+we|let\s*s|lets)\s+(?:mag-?)?english\b",
        r"\b(?:i\s+want\s+to|i\s+wanna|id\s+like\s+to|want\s+to|wanna)\s+(?:practice|learn|study|speak|talk\s+in)\s+(?:in\s+)?english\b",
        r"\b(?:help\s+me\s+with|help\s+me\s+learn|help\s+me\s+speak)\s+english\b",
        r"\b(?:how\s+to\s+speak|teach\s+me\s+how\s+to\s+speak)\s+english\b",
        r"\bmag-?usap\s+tayo\s+sa\s+english\b",
    ]
    if any(re.search(p, low) for p in english_patterns):
        return "en"

    return ""

def _detect_language(raw: str, history: list[dict] = None) -> str:
    # 1. Direct explicit language shift request in current user message
    shift = _check_language_shift(raw)
    if shift:
        return shift

    # 2. Check recent conversation history for an active language shift preference
    if history:
        for msg in reversed(history[:-1]):
            if msg.get("role") == "user":
                prev_shift = _check_language_shift(msg.get("content", ""))
                if prev_shift:
                    return prev_shift

    words = re.findall(r"[a-zA-ZÀ-ÿ']+", raw.lower())
    if not words:
        return "en"
    fil_words = {
        "ang", "mga", "sa", "ng", "na", "ko", "mo", "ako", "ikaw", "ka", "siya", "kami", "tayo", "kayo", "sila",
        "po", "opo", "ho", "ba", "pa", "naman", "lang", "lamang", "kasi", "pero", "kung", "kapag", "dahil", "para",
        "ano", "anong", "bakit", "paano", "saan", "kailan", "sino", "kumusta", "kamusta", "kwento", "kuwento",
        "magkwento", "kuwentuhan", "magjoke", "biro", "oo", "hindi", "wag", "huwag", "wala", "walang", "meron",
        "may", "masaya", "malungkot", "pagod", "gutom", "antok", "puyat", "kain", "kumakain", "kumain", "ulam",
        "aral", "laro", "ituloy", "tuloy", "dugtungan", "sige", "talaga", "grabe", "uy", "lods", "pre", "pare",
        "tol", "bes", "beh", "mare", "namin", "natin", "nila", "akin", "iyo", "kanila", "ito", "iyan", "iyon",
        "dito", "diyan", "doon", "ngayon", "bukas", "kahapon", "kanina", "mamaya", "araw", "gabi", "umaga",
        "hapon", "magandang", "gusto", "ayaw", "mahilig", "mahal", "bahay", "trabaho", "eskwela", "klase",
        "kaibigan", "pamilya", "tropa", "barkada", "ulan", "maulan", "init", "mainit", "lamig", "malamig",
        "salamat", "pasensya", "pwede", "dapat", "sana", "yung", "ung", "yan", "yon", "ganun", "ganyan",
    }
    en_words = {
        "the", "a", "an", "is", "are", "am", "was", "were", "be", "been", "being", "what", "why", "how", "when",
        "where", "who", "which", "can", "could", "would", "should", "will", "shall", "may", "might", "must",
        "do", "does", "did", "done", "doing", "have", "has", "had", "having", "tell", "story", "joke", "about",
        "you", "your", "yours", "my", "mine", "me", "we", "us", "our", "they", "them", "their", "he", "him",
        "his", "she", "her", "it", "its", "this", "that", "these", "those", "want", "need", "like", "love",
        "feel", "feeling", "think", "thinking", "know", "good", "great", "nice", "today", "tomorrow", "yesterday",
        "now", "please", "continue", "more", "i", "i'm", "im", "i've", "dont", "don't", "cant", "can't", "wont",
        "won't", "and", "or", "but", "because", "if", "so", "for", "with", "from", "to", "in", "on", "at", "by",
        "work", "working", "study", "studying", "eat", "eating", "play", "playing", "watch", "watching", "tired",
        "hungry", "happy", "sad", "hello", "hi", "hey", "thanks", "thank",
    }
    fil_count = sum(1 for w in words if w in fil_words or (w not in en_words and len(w) > 5 and w.startswith(("nag", "mag", "na", "ma", "pa", "ka", "um", "in", "pag", "pinag", "nakaka"))))
    en_count = sum(1 for w in words if w in en_words)
    if fil_count > 0 and en_count == 0:
        return "fil"
    if en_count > 0 and fil_count == 0:
        return "en"
    if fil_count > 0 and en_count > 0:
        if fil_count >= en_count * 2.5:
            return "fil"
        if en_count >= fil_count * 2.5:
            return "en"
        return "both"
    return "en"

def _detect_tutor_tip(raw: str, mode: str) -> str:
    low = raw.lower()

    # 1. Double past tense / "did" + past tense (common Philippine English slip)
    m_did = re.search(r"\b(did(?:n't|\s+not)?)\s+(went|ate|saw|took|bought|came|told|found|made|gave|knew|got|spoke|wrote|left|lost|felt)\b", low)
    if m_did:
        did_word, past_v = m_did.group(1), m_did.group(2)
        base_map = {
            "went": "go", "ate": "eat", "saw": "see", "took": "take", "bought": "buy",
            "came": "come", "told": "tell", "found": "find", "made": "make", "gave": "give",
            "knew": "know", "got": "get", "spoke": "speak", "wrote": "write", "left": "leave",
            "lost": "lose", "felt": "feel"
        }
        base_v = base_map.get(past_v, "base form")
        return f"💡 Quick Tip: In English, after auxiliary 'did' or 'didn\\'t', use the base form of the verb. Instead of *\"{did_word} {past_v}\"*, say **\"{did_word} {base_v}\"** (or simply **\"{past_v}\"** without 'did')."

    # 2. "Cope up with"
    if "cope up with" in low or "coping up with" in low or "cope up" in low:
        return "💡 Quick Tip: In standard English, the idiom is **\"cope with\"** rather than *\"cope up with\"* (e.g., *\"I'm learning to cope with the busy workload\"*)."

    # 3. "Fill up" for forms/documents
    if re.search(r"\b(?:fill|filled|filling)\s+up\s+(?:the\s+|a\s+|this\s+|that\s+|my\s+|our\s+)?(?:form|forms|application|survey|document|sheet)\b", low):
        return "💡 Quick Tip: Native English speakers usually say **\"fill out\"** (or **\"fill in\"**) a form. *\"Fill up\"* is normally used for liquids and containers (e.g., *\"fill up the glass\"*)."

    # 4. "Open / close the light"
    if re.search(r"\b(?:open|opened|opening|close|closed|closing)\s+the\s+light\b", low):
        return "💡 Quick Tip: In English, lights are switched, not opened. Say **\"turn on / switch on the light\"** or **\"turn off / switch off the light\"** instead of *\"open / close the light\"*."

    # 5. Uncountable nouns with plural 's': informations, feedbacks, advices, equipments, stuffs
    m_uncount = re.search(r"\b(informations|feedbacks|advices|equipments|stuffs)\b", low)
    if m_uncount:
        word = m_uncount.group(1)
        corrected = {
            "informations": "information (or 'pieces of information')",
            "feedbacks": "feedback (or 'some feedback')",
            "advices": "advice (or 'words of advice')",
            "equipments": "equipment",
            "stuffs": "stuff (or 'things')",
        }.get(word, word)
        return f"💡 Quick Tip: *\"{word}\"* is an uncountable (mass) noun in English, so we don't add an '-s'. Use **\"{corrected}\"** instead."

    # 6. "I am agree" / "I'm agree"
    if re.search(r"\bi\s*(?:am|'m)\s+agree\b", low):
        return "💡 Quick Tip: 'Agree' is already a verb on its own! Instead of *\"I am agree\"*, simply say **\"I agree\"** or **\"I agree with you\"**."

    # 7. Subject-verb agreement: "he don't", "she don't", "it don't"
    m_agr = re.search(r"\b(he|she|it)\s+don't\b", low)
    if m_agr:
        subj = m_agr.group(1)
        return f"💡 Quick Tip: For third-person singular (*{subj}*), remember to use **\"{subj} doesn't\"** instead of *\"{subj} don't\"* in standard English."

    # 8. Redundant prepositions: "discuss about", "mention about"
    if re.search(r"\b(?:discuss|discussed|discussing)\s+about\b", low):
        return "💡 Quick Tip: 'Discuss' already implies 'talking about' something. You can simply say **\"discuss the topic\"** instead of *\"discuss about\"*."
    if re.search(r"\b(?:mention|mentioned|mentioning)\s+about\b", low):
        return "💡 Quick Tip: 'Mention' directly takes the object without 'about'—for example, **\"mention the idea\"** rather than *\"mention about\"*."

    # 9. Tagalog learner errors:
    # "Ako gusto" -> "Gusto ko"
    if re.search(r"\bako\s+(?:ay\s+)?gusto\b", low):
        return "✨ Filipino Tip: In conversational Tagalog, predicate-first word order sounds much more natural! Instead of *\"Ako gusto...\"*, say **\"Gusto ko...\"** (e.g., *\"Gusto kong mag-aral\"*)."

    # "Gusto ko ikaw" -> "Gusto kita"
    if re.search(r"\bgusto\s+ko\s+ikaw\b", low):
        return "✨ Filipino Tip: In Tagalog, when you like or want someone, the pronouns combine into **\"kita\"**: say **\"Gusto kita\"** instead of *\"gusto ko ikaw\"*."

    # "Ako mahal kita" / "I mahal you" -> "Mahal kita"
    if re.search(r"\b(?:ako\s+mahal\s+kita|i\s+mahal\s+you)\b", low):
        return "✨ Filipino Tip: In Tagalog, the verb comes first and the pronouns are already inside 'kita': simply say **\"Mahal kita\"** (I love you)!"

    # "Nagkain ako" -> "Kumain ako"
    if re.search(r"\bnagkain\s+ako\b", low):
        return "✨ Filipino Tip: For the root word *kain* (eat), the standard past tense uses the -um- infix: **\"Kumain ako\"** rather than *\"nagkain ako\"*."

    # "Saan ang..." for static location -> "Nasaan ang..."
    if re.search(r"\bsaan\s+ang\b", low):
        return "✨ Filipino Tip: When asking where a specific object or place is located right now, **\"Nasaan ang...\"** is more precise than *\"Saan ang...\"* (*Saan* is typically used for destinations, like *Saan ka pupunta?*)."

    # Adjective without linker: "maganda babae", "masarap pagkain"
    m_adj = re.search(r"\b(maganda|masarap|mabait|mabilis|matalino)\s+(babae|lalaki|pagkain|tao|bata|bahay)\b", low)
    if m_adj:
        adj, noun = m_adj.group(1), m_adj.group(2)
        linker = "ng" if adj.endswith(('a', 'e', 'i', 'o', 'u')) else " na"
        return f"✨ Filipino Tip: In Tagalog, adjectives need a linker (**-ng** or **na**) before the noun they describe: instead of *\"{adj} {noun}\"*, say **\"{adj}{linker} {noun}\"**."

    return ""

def _build_local_reply(mode: str, history: list[dict]) -> str:
    raw = history[-1]["content"].strip()
    shift = _check_language_shift(raw)
    if shift:
        mode = shift
    elif mode == "auto":
        mode = _detect_language(raw, history)
    msg = raw.lower()
    style = _extract_style(raw, history)
    po = " po" if style["polite"] and mode != "en" else ""
    addr = f" {style['slang']}" if style['slang'] else (f" {style['name']}" if style['name'] else "")
    haha = " 😂" if style["playful"] else ""
    reflected = _reflect_user_text(raw, mode)
    prev_bot = [m["content"] for m in history[:-1] if m["role"] == "assistant"]

    def _ret(text: str, skip_tip: bool = False) -> str:
        if skip_tip:
            return text
        tip = _detect_tutor_tip(raw, mode)
        return f"{text}\n\n{tip}" if tip else text

    # 1. Safety / sadness / distress
    if any(w in msg for w in ("sad", "malungkot", "lonely", "hopeless", "pagod na ako sa buhay", "depressed", "iyak", "crying", "heartbroken", "brokenhearted")):
        if mode == "fil":
            return _ret(f"Naku{addr}, ramdam ko{po} na mabigat ang pinagdadaanan mo ngayong sinabi mong '{raw}'. Nandito lang ako{po} para makinig sa'yo. Huwag ka{po} sanang mag-atubiling lumapit sa pamilya o kaibigang pinagkakatiwalaan mo—hindi ka nag-iisa. Gusto mo bang ikuwento pa kung ano ang nangyari?", skip_tip=True)
        if mode == "en":
            return _ret(f"I'm really sorry to hear that{addr}—it sounds heavy when you say '{raw}'. I'm right here to listen, and please consider reaching out to a trusted friend, family member, or support line so you don't have to carry this alone. Do you want to talk more about what happened?", skip_tip=True)
        return _ret(f"Uy{addr}, I'm really sorry to hear na ganyan ang nararamdaman mo{po} ngayon—lalo na sa sinabi mong '{raw}'. Nandito lang ako{po} to listen, and please don't hesitate to reach out sa family o trusted friends mo ha. Gusto mo bang i-share kung ano'ng nangyari?", skip_tip=True)

    # 2. Pure Language Shift Greeting (e.g. "I want to learn Tagalog", "English please", "Tagalog", "Mag-Tagalog tayo")
    is_lesson_request = any(p in msg for p in (
        "teach", "basics", "difference", "explain", "order", "dining", "commute", "particles",
        "filipinism", "how to say", "how do you say", "question words", "verb prefixes", "conjugation",
        "frequency", "comparing", "opinions", "idioms", "slang", "numbers", "money", "greetings",
        "feelings", "health", "family", "weather", "chores", "emergency", "workplace"
    ))
    if shift and not is_lesson_request:
        if shift == "fil":
            return _ret(
                f"Galing! Tara, mag-Tagalog tayo{po}{addr}! 🎉 Handa na akong tulungan ka sa pag-aaral ng Filipino. "
                f"Kumusta ka ngayon? Subukan mong sabihin sa akin sa Tagalog kung kumusta ang araw mo o ano ang gusto mong pag-usapan!"
            )
        if shift == "en":
            return _ret(
                f"Awesome, let's practice English{addr}! 🎉 You've got this, and I'll gently guide you with tips along the way. "
                f"How has your day been going so far? Tell me what you've been up to today!"
            )
        if shift == "both":
            return _ret(
                f"Ayun, let's do Taglish{po}{addr}! 🎉 Best of both worlds—natural code-switching and conversational practice. "
                f"Kumusta naman ang araw mo today? Tell me what you've been up to lately!"
            )

    # 3. Dedicated Beginner Lessons & Basics Curriculum (e.g. from suggestion buttons)
    # 3a. Starter Kit & General Basics Overview (with literal comparison boxes)
    if any(p in msg for p in (
        "teach me the basics", "teach me basics", "basics of tagalog and english", "basics of tagalog",
        "basics of english", "basic grammar", "parts of speech", "learn basics", "starter kit",
        "starter", "boxes that shows the difference", "improve the lines and coolumn", "improve the lines and column",
        "literal box", "literal boxs"
    )):
        return _ret(
            f"Sure thing{addr}! Let’s start with a **starter kit** for both Tagalog and English with clear comparison boxes so you can see the differences and similarities easily. Ready? 🚀\n\n"
            f"---\n\n"
            f"## 1️⃣ Tagalog Basics\n\n"
            f"| Category | Example | What It Means |\n"
            f"|---|---|---|\n"
            f"| Greetings | *Kamusta? / Kumusta?* | “How are you?” |\n"
            f"| Pronouns | *Ako (I), Ikaw (you), Siya (he/she), Kami (we), Kayo (you plural), Sila (they)* | The subject that comes BEFORE the verb. |\n"
            f"| Basic Structure | *Kumain ako ng kanin.* | “I ate rice.” – Verb‑Subject‑Object. |\n"
            f"| Common Verbs | *Kumain (eat), Uminom (drink), Pumunta (go), Bumili (buy)* | Simple “doing” words. |\n"
            f"| Adjectives | *Maganda (pretty), Masarap (delicious)* | Usually come *after* the noun. |\n"
            f"| Numbers | *Isa (1), Dalawa (2), Tatlo (3)* | Counting. |\n"
            f"| Phrase to Start a Sentence | *Sa tingin ko,* | “I think that…” (nice for opinion). |\n\n"
            f"### Practice Prompt (Tagalog)\n"
            f"Write a one‑sentence story about *what you did yesterday* using the pattern:\n"
            f"**Verb + Subject + Object.**\n"
            f"Example: *Nagluto ako ng kanin.*\n\n"
            f"---\n\n"
            f"## 2️⃣ English Basics\n\n"
            f"| Category | Example | What It Means |\n"
            f"|---|---|---|\n"
            f"| Greetings | *Hi there! / How are you?* | “Hello” and “How’s it going?” |\n"
            f"| Pronouns | *I, You, He, She, It, We, They* | The subject (or object) that comes *before* the verb. |\n"
            f"| Basic Structure | *I ate rice.* | Subject‑Verb‑Object. |\n"
            f"| Common Verbs | *Eat, Drink, Go, Buy* | The same ideas as in Tagalog. |\n"
            f"| Adjectives | *pretty, yummy* | Usually come *before* the noun. |\n"
            f"| Numbers | *one, two, three* | Counting. |\n"
            f"| Phrase to Start a Sentence | *In my opinion,* | Nice for giving opinions. |\n\n"
            f"### Practice Prompt (English)\n"
            f"Write a one‑sentence story about *what you did yesterday* using the pattern:\n"
            f"**Subject + Verb + Object.**\n"
            f"Example: *I ate rice.*\n\n"
            f"---\n\n"
            f"## 3️⃣ Taglish: Mixing Both\n\n"
            f"| Tagalog | English | Taglish |\n"
            f"|---|---|---|\n"
            f"| *Kumain ako ng kanin.* | *I ate rice.* | *I kumain ng kanin.* |\n\n"
            f"💡 **Why do people mix?**\n"
            f"• Many Filipinos code‑switch in daily chat or when speaking in groups.\n"
            f"• You can keep the verb in one language and mix the rest for fun, but practice staying consistent first!\n\n"
            f"---\n\n"
            f"### Your Turn! 💬\n\n"
            f"1. **Tagalog sentence:** “I walked to the market.”\n"
            f"   Write it with the Tagalog structure: *Naglakad ako ___?*\n\n"
            f"2. **English sentence:** “I bought bread.”\n"
            f"   Just write it in plain English.\n\n"
            f"3. **Taglish sentence:** Mix one Tagalog word into each:\n"
            f"   *I ___ yesterday at the market.* (Choose a Tagalog verb that means “walk”.)\n\n"
            f"Feel free to write your answers below! I’ll give you gentle feedback and quick tips. 🌟"
        )

    # 3a-2. Mini-Class: Grammar Refresher, Dialogue & Exercise (with Tagalog & Taglish Translations)
    if any(p in msg for p in ("mini class", "mini-class", "grammar refresher", "grammar exercise", "quick grammar", "practice dialogue", "quick grammar refresher")):
        return _ret(
            f"Sure thing{po}{addr}! 😄 Let’s set up a quick **mini‑class** so you can practice speaking and get a feel for proper English and Tagalog grammar. "
            f"Here is the lesson complete with translations and practice prompts! 🌟\n\n"
            f"---\n\n"
            f"## 📚 1️⃣ Quick Grammar Refresher\n\n"
            f"| Element | Tagalog Example | English Example | Why It Matters |\n"
            f"|---|---|---|---|\n"
            f"| Subject‑Verb‑Object (SVO) | *Ako ay kumain ng kanin.* | *I ate rice.* | Most English sentences go Subject → Verb → Object. |\n"
            f"| Pronoun Usage | *Ikaw ang bida.* | *You’re the star.* | In English, the pronoun (I, you, he, she, we, they) always comes before the verb. |\n"
            f"| Present Simple | *Nag-aaral siya araw-araw.* | *He studies every day.* | Use for regular habits, routines, or daily facts. |\n"
            f"| Past Simple | *Tumawa siya.* | *He laughed.* | Use *-ed* ending for regular verbs (laugh → laughed). |\n"
            f"| Future with \"will\" | *Dadaan ako bukas.* | *I will see you / visit tomorrow.* | \"Will\" marks actions happening in the future. |\n\n"
            f"💡 **Quick Tip:** When you’re not sure how to say something in English, start with the simplest structure: **Subject + Verb + Object**. Then, add details.\n\n"
            f"---\n\n"
            f"## 🎤 2️⃣ Practice Dialogue\n\n"
            f"• **You (English speaker):**\n"
            f"  > *\"Hi! I’m learning English.\"*\n"
            f"  > *(Tagalog: Hi! Nag-aaral ako ng Ingles.)*\n\n"
            f"• **Me (Taglish Tutor):**\n"
            f"  > *\"That’s awesome! What made you want to study English?\"*\n"
            f"  > *(Tagalog: Ang galing naman! Bakit mo gustong mag-aral ng Ingles?)*\n\n"
            f"👉 **Your Turn:** Reply in English below!\n"
            f"> *What made you want to study English?*\n"
            f"*(Tagalog: Ano ang dahilan kung bakit gusto mong mag-aral ng Ingles?)*\n\n"
            f"---\n\n"
            f"## 🏆 3️⃣ Quick Grammar Exercise\n\n"
            f"| # | Sentence (English) | Tagalog Translation | Casual Taglish Alternative | Verb Tense |\n"
            f"|---|---|---|---|---|\n"
            f"| 1 | *She walks to school every day.* | *Naglalakad siya papuntang paaralan araw-araw.* | *Naglalakad siya papasok sa school everyday.* | Present Simple |\n"
            f"| 2 | *They ate pizza yesterday.* | *Kumain sila ng pizza kahapon.* | *Kumain sila ng pizza kahapon.* | Past Simple |\n"
            f"| 3 | *I will visit my grandma next week.* | *Bibisitahin ko si lola sa susunod na linggo.* | *Pupuntahan ko si lola next week.* | Future Simple |\n\n"
            f"---\n\n"
            f"### 💡 Quick Tips for Daily Practice\n\n"
            f"• **Use 'I' or 'You' first:** Those are your sentence subjects.\n"
            f"• **Verb endings:** *-ed* for regular past tense (*walk → walked*), irregular verbs change (*eat → ate*).\n"
            f"• **Ask questions:** Add *\"What\"* (Ano) or *\"Why\"* (Bakit) to practice conversational prompts.\n\n"
            f"Whenever you make a mistake, just write back and I’ll gently help you out. Galing! 🚀"
        )

    # Apologies & Levels of Formality (Matching Reference Photo)
    if any(p in msg for p in ("apolog", "sorry", "my bad", "excuse me", "humingi ng pasensya", "patawad", "pardon")):
        return _ret(
            f"Here is how to apologize or excuse yourself across different levels of formality{po}{addr}! 🙏\n\n"
            f"| Level | English | Tagalog / Taglish |\n"
            f"|---|---|---|\n"
            f"| Casual \"My bad\" | \"Sorry about that!\" | \"Sorry talaga!\" / \"Sensya na!\" |\n"
            f"| Formal apology | \"I sincerely apologize.\" | \"Humingi po ako ng pasensya / tawad.\" |\n"
            f"| Accidental mistake | \"I didn't mean to do that.\" | \"Hindi ko sinasadya.\" |\n"
            f"| Excuse me (passing by) | \"Excuse me, coming through.\" | \"Makikiraan po.\" |\n"
            f"| Forgiving someone | \"No problem / It's okay.\" | \"Okay lang 'yun!\" / \"Wala 'yun!\" |\n\n"
            f"💡 **Practice Challenge:** Imagine you accidentally bumped into someone at a busy train station or mall. Which phrase would you say?"
        )

    # 3b. Tagalog Basics & Greetings
    if any(p in msg for p in ("tagalog basics", "basic tagalog conversation", "basic tagalog greetings", "tagalog conversation and greetings")):
        return _ret(
            f"Tara, let's learn **Basic Tagalog Conversation**{po}{addr}! 🇵🇭\n\n"
            f"| Situation | Tagalog Phrase | English Meaning |\n"
            f"|---|---|---|\n"
            f"| Morning Greeting | *Magandang umaga po* | Good morning |\n"
            f"| Afternoon Greeting | *Magandang hapon po* | Good afternoon |\n"
            f"| Evening Greeting | *Magandang gabi po* | Good evening |\n"
            f"| Asking how they are | *Kumusta ka?* | How are you? |\n"
            f"| Saying Thank you | *Salamat po* | Thank you |\n"
            f"| You're welcome | *Walang anuman po* | You're welcome |\n"
            f"| Expressing desire | *Gusto ko ng...* | I want [something] |\n"
            f"| Asking location | *Nasaan ang...?* | Where is [something]? |\n\n"
            f"💡 **Your Turn:** Try greeting me and saying what you want today (e.g. *\"Magandang araw po! Gusto ko ng kape.\"*)!"
        )

    # 3c. English Basics
    if any(p in msg for p in ("english basics", "basic english speaking", "basic english grammar", "help me practice basic english")):
        return _ret(
            f"Awesome, let's practice **Basic English Speaking**{addr}! 🇬🇧\n\n"
            f"| Core Rule | Formula | Example Sentence |\n"
            f"|---|---|---|\n"
            f"| Sentence Order | **Subject + Verb + Object** | *I am drinking water.* |\n"
            f"| Past Tense | Completed action | *I ate lunch already.* |\n"
            f"| Present Tense | Happening right now | *I am eating lunch right now.* |\n"
            f"| Future Tense | Plan or intention | *I will eat lunch later.* |\n\n"
            f"💡 **Practice Challenge:** Tell me one thing you did earlier today in a full English sentence!"
        )

    # 3d. The 3 Types of "No": Hindi vs. Ayaw vs. Huwag
    if any(p in msg for p in ("hindi vs", "ayaw vs", "huwag vs", "difference between hindi", "types of no")):
        return _ret(
            f"Great topic{po}{addr}! In Tagalog, there are **3 different ways to say 'No'**, and each has a specific role:\n\n"
            f"| Type of \"No\" | Core Meaning & Role | Example Sentence | English Translation |\n"
            f"|---|---|---|---|\n"
            f"| **Hindi** | Facts & Negation | *Hindi ako pagod.* | I am not tired. |\n"
            f"| **Ayaw** | Personal Dislike & Refusal | *Ayaw ko ng maanghang.* | I don't want spicy food. |\n"
            f"| **Huwag** | Command & Prohibition | *Huwag kang mag-alala!* | Don't worry! |\n\n"
            f"💡 **Quick Quiz:** If you want to say *\"Don't forget!\"*, would you say **\"Hindi mo kalimutan\"** or **\"Huwag mong kalimutan\"**? Try answering!"
        )

    # 3e. Dining & Ordering Food
    if any(p in msg for p in ("order food", "ordering food", "dining phrases", "mag-order ng pagkain", "restaurant phrases")):
        return _ret(
            f"Let's practice **Dining & Ordering Food**{po}{addr}! 🍽️\n\n"
            f"| Situation | Tagalog / Taglish | Natural English Alternative |\n"
            f"|---|---|---|\n"
            f"| Asking for menu | *\"Pahingi po ng menu.\"* | \"Could I see the menu, please?\" |\n"
            f"| Ordering drinks | *\"Isa pong iced coffee, please.\"* | \"One iced coffee, please.\" |\n"
            f"| Requesting the bill | *\"Pa-bill out po!\"* | \"Could we get the bill/check, please?\" |\n"
            f"| Takeout leftovers | *\"Pa-balot po nito.\"* | \"Could you box this up for takeout?\" |\n\n"
            f"💡 **Roleplay Practice:** Imagine we're at a cozy cafe in Manila or New York! How would you order your favorite drink or ask for the bill?"
        )

    # 3f. Commuting & Travel
    if any(p in msg for p in ("commuting", "commute phrases", "jeepney", "riding jeepneys", "para po", "bayad po")):
        return _ret(
            f"Tara, alamin natin ang **Commuting Survival Phrases**{po}{addr}! 🚌\n\n"
            f"| Action / Step | Say This (Tagalog) | English Meaning |\n"
            f"|---|---|---|\n"
            f"| Paying fare | *\"Bayad po!\"* | Here is my fare! |\n"
            f"| Passing fare | *\"Pa-abot naman po ng bayad, salamat!\"* | Please pass my fare forward! |\n"
            f"| Asking for change | *\"Sukli po nung bente/singkwenta?\"* | Change for my 20/50, please? |\n"
            f"| Requesting to stop | *\"Para po!\"* / *\"Sa tabi lang po!\"* | Pull over / Drop me off here! |\n"
            f"| Checking route | *\"Dadaan po ba 'to sa Cubao?\"* | Does this pass through Cubao? |\n\n"
            f"💡 **Practice Challenge:** How would you tell a driver that you want to get off at the corner?"
        )

    # 3g. Taglish Particles (naman, pala, daw, ba)
    if any(p in msg for p in ("taglish particles", "particles", "naman pala daw", "na pa naman")):
        return _ret(
            f"These little words are the secret sauce of natural Filipino & Taglish speech{po}{addr}! 💬\n\n"
            f"| Particle | Nuance / Meaning | Example Sentence | English Translation |\n"
            f"|---|---|---|---|\n"
            f"| **Na** | Already / Now | *Tapos na ako.* | I am done already. |\n"
            f"| **Pa** | Still / Yet / More | *Kumakain pa ako.* | I am still eating. |\n"
            f"| **Naman** | Softener / Turn-taking | *Ikaw naman.* / *Sige na naman.* | Your turn! / Please (softened). |\n"
            f"| **Pala** | Realization / Surprise | *Ikaw pala!* | Oh, it's you! (Didn't realize!) |\n"
            f"| **Daw / Raw** | Hearsay / 'They said' | *Uulan daw mamaya.* | They say it will rain later. |\n"
            f"| **Ba** | Question marker | *Masarap ba?* | Is it delicious? |\n\n"
            f"💡 **Your Turn:** Try making a sentence using **'pala'** or **'naman'**!"
        )

    # 3h. Filipinisms to Polish
    if any(p in msg for p in ("filipinism", "filipinisms", "polish filipinism")):
        return _ret(
            f"Here are the top **Filipinisms to polish** for natural, international English{addr}! ✨\n\n"
            f"| Avoid (Direct Translation) | Natural English Alternative | Why / Natural Context |\n"
            f"|---|---|---|\n"
            f"| ❌ *\"Open the light\"* | ✅ *\"Turn on the light\"* | Lights operate with switches, not hinges. |\n"
            f"| ❌ *\"Close the computer\"* | ✅ *\"Turn off / Shut down the PC\"* | 'Close' sounds like folding the physical laptop lid. |\n"
            f"| ❌ *\"I will go ahead\"* | ✅ *\"I'm heading out / leaving now\"* | 'Go ahead' in English means giving permission. |\n"
            f"| ❌ *\"Fill up this form\"* | ✅ *\"Fill out this form\"* | In international English, documents are 'filled out'. |\n"
            f"| ❌ *\"I will take you home\"* | ✅ *\"I'll drive you home / drop you off\"* | Sounds more natural when offering someone a ride. |\n\n"
            f"💡 **Practice Challenge:** Pick one of these phrases and use the natural English version in a sentence!"
        )

    # 3i. Question Words (Mga Salitang Pananong)
    if any(p in msg for p in ("question words", "mga salitang pananong", "sino ano saan", "sino", "alin", "kanino")):
        return _ret(
            f"Asking questions is the quickest way to practice conversation{po}{addr}! ❓\n\n"
            f"| In Tagalog | In English | What It Means / Example Use |\n"
            f"|---|---|---|\n"
            f"| **Sino** | Who | *“Sino ang naglalakad?”* → Who is walking? |\n"
            f"| **Ano / Ano ang** | What | *“Ano ang ginagawa mo?”* → What are you doing? |\n"
            f"| **Kailan** | When | *“Kailan ka aalis?”* → When are you leaving? |\n"
            f"| **Saan** | Where | *“Saan ang banyo?”* → Where is the bathroom? |\n"
            f"| **Bakit** | Why | *“Bakit ka malungkot?”* → Why are you sad? |\n"
            f"| **Paano** | How | *“Paano mo ginagawa ‘yan?”* → How do you do that? |\n"
            f"| **Alin / Alin sa** | Which (among a list) | *“Alin sa dalawang ito?”* → Which of these two? |\n"
            f"| **Kanino** | Whose | *“Kanino ang bangang 'yan?”* → Whose pot was that? |\n\n"
            f"> 💡 **Tips on Tagalog Question Words:**\n"
            f"> 1. Use **“Ano”** when you’re asking about a thing or general feelings (*Ano ang pakiramdam mo?*).\n"
            f"> 2. Use **“Ano ang”** literally *“What is the…”* for the same effect.\n"
            f"> 3. **“Alin”** is only for picking between two or a specific list; use **“Ano”** for broader options.\n\n"
            f"💡 **Practice Challenge:** Try asking me any question in Tagalog or English using one of these question words!"
        )

    # 3j. Taglish Verb Conjugation
    if any(p in msg for p in ("verb prefixes", "taglish verb", "conjugation", "mag- nag-", "mag nag")):
        return _ret(
            f"Here is how Filipinos naturally blend English verbs with Tagalog prefixes{po}{addr}! 🔄\n\n"
            f"| Aspect / Tense | Tagalog Prefix Pattern | Example Sentence | English Meaning |\n"
            f"|---|---|---|---|\n"
            f"| **Past** (Completed) | **Nag-** + verb | *Nag-search ako kanina.* | I searched earlier. |\n"
            f"| **Present** (Ongoing) | **Nagse-** / syllable repeat | *Nagse-search pa ako.* | I am still searching right now. |\n"
            f"| **Future** (Contemplated) | **Magse-** + verb | *Magse-search ako mamaya.* | I will search later. |\n"
            f"| **Accidental / Passive** | **Na-** + verb | *Na-delete ko 'yung file!* | I accidentally deleted the file! |\n"
            f"| **Polite Request** | **Paki-** + verb | *Paki-send naman sa email.* | Could you please send it to my email? |\n\n"
            f"💡 **Practice Challenge:** Pick an English verb (like *download*, *cook*, or *call*) and try turning it into a Taglish sentence!"
        )

    # 3k. Time, Frequency, & Routine
    if any(p in msg for p in ("time and frequency", "frequency", "dalas at panahon", "palagi madalas")):
        return _ret(
            f"Here is the **Frequency Scale & Time Markers** for daily routines{po}{addr}! ⏰\n\n"
            f"| Frequency | Tagalog Term | English Meaning | Example Sentence |\n"
            f"|---|---|---|---|\n"
            f"| 100% | **Palagi / Lagi** | Always | *Palagi siyang maaga.* (Always early) |\n"
            f"| 80% | **Madalas** | Often / Usually | *Madalas akong mag-kape.* (Often drink coffee) |\n"
            f"| 50% | **Minsan** | Sometimes | *Minsan lang kami lumabas.* (Sometimes go out) |\n"
            f"| 20% | **Bihira** | Rarely / Seldom | *Bihira siyang magpuyat.* (Rarely stays up late) |\n"
            f"| 0% | **Hindi kailanman** | Never | *Hindi pa ako nakapunta doon.* (Never been there) |\n\n"
            f"💡 **Practice Challenge:** Tell me one thing you do *often* (**madalas**) and one thing you do *rarely* (**bihira**)!"
        )

    # 3l. Comparing Things (Comparatives)
    if any(p in msg for p in ("comparing things", "paghahambing", "mas pinaka", "kasing")):
        return _ret(
            f"Here is how to compare things easily in Tagalog and English{po}{addr}! ⚖️\n\n"
            f"1. **More / -er: Use 'Mas'**\n"
            f"   • *\"Mas mabilis ang bagon kaysa kotse.\"* = The train is faster than the car.\n"
            f"   • *\"Mas masarap ang lutong bahay.\"* = Home-cooked food is more delicious.\n\n"
            f"2. **Most / -est: Use 'Pinaka-'**\n"
            f"   • *\"Ito ang pinakamurang bilihin.\"* = This is the cheapest item.\n"
            f"   • *\"Siya ang pinakamabait.\"* = He/She is the kindest.\n\n"
            f"3. **Equal / As...as: Use 'Kasing-' / 'Pareho'**\n"
            f"   • *\"Kasingtangkad mo siya.\"* = You are as tall as him/her.\n\n"
            f"💡 **Practice Challenge:** Compare two things you like (e.g. coffee vs. tea, or beach vs. mountains)!"
        )

    # 3m. Giving Opinions & Disagreeing Politely
    if any(p in msg for p in ("giving opinions", "disagreeing politely", "sa tingin ko", "in my opinion")):
        return _ret(
            f"Here are the smoothest ways to share opinions and disagree with diplomacy{po}{addr}! 🗣️\n\n"
            f"• **Stating an Opinion:**\n"
            f"  - Tagalog: *\"Sa tingin ko...\"* / *\"Sa palagay ko...\"*\n"
            f"  - English: *\"In my opinion...\"* / *\"I think that...\"*\n\n"
            f"• **Agreeing Enthusiastically:**\n"
            f"  - Tagalog: *\"Sang-ayon ako sa 'yo!\"* / *\"Tama ka diyan!\"*\n"
            f"  - English: *\"I completely agree!\"* / *\"Exactly!\"*\n\n"
            f"• **Gentle / Diplomatic Disagreement:**\n"
            f"  - Taglish: *\"Okay naman 'yung point mo, pero baka pwede nating i-consider...\"*\n"
            f"  - English: *\"I see your point, but on the other hand...\"*\n\n"
            f"💡 **Practice Challenge:** What is your opinion on working from home versus going to the office? Try answering using one of these phrases!"
        )

    # 3n. Idioms & Slang
    if any(p in msg for p in ("idioms", "slang", "balat-sibuyas", "petiks", "under the weather")):
        return _ret(
            f"Idioms add flavor and humor to conversation because their meanings go beyond the words{po}{addr}! 🎭\n\n"
            f"• **Famous Filipino Idioms:**\n"
            f"  - **Balat-sibuyas** (literally *onion-skinned*): Overly sensitive or easily hurt.\n"
            f"  - **Ningas-kugon** (literally *cogon grass fire*): Starting with blazing enthusiasm that burns out fast.\n"
            f"  - **Petiks**: Relaxing, chilling, or taking it easy at work.\n\n"
            f"• **Essential English Idioms:**\n"
            f"  - **Under the weather**: Feeling a bit sick or unwell (*Medyo masama ang pakiramdam*).\n"
            f"  - **Piece of cake**: Super easy! (*Sisiw lang / Napakadali*).\n"
            f"  - **Cost an arm and a leg**: Extremely expensive! (*Napakmahal / Ginto ang presyo*).\n\n"
            f"💡 **Your Turn:** Have you ever experienced a day that felt like a 'piece of cake', or felt 'under the weather'?"
        )

    # 3o. Numbers, Money, & Telling Time
    if any(p in msg for p in ("numbers", "money", "telling time", "oras at pera", "magkano")):
        return _ret(
            f"In the Philippines, numbers are uniquely blended across Tagalog, Spanish, and English{po}{addr}! 💵\n\n"
            f"• **Counting Objects (Use Tagalog):**\n"
            f"  - *Isa* (1), *dalawa* (2), *tatlo* (3), *apat* (4), *lima* (5), *sampu* (10).\n"
            f"  - Example: *Dalawang kape* (2 coffees).\n\n"
            f"• **Prices & Telling Time (Use Spanish loanwords or English):**\n"
            f"  - Prices: *Bente pesos* (₱20), *singkwenta* (₱50), *isang daan* (₱100).\n"
            f"  - Time: *Ala-una* (1:00), *Alas-dos* (2:00), *Alas-tres ng hapon* (3:00 PM).\n\n"
            f"💡 **Quick Challenge:** Try telling me what time it is right now in your area, or ask how much a coffee costs!"
        )

    # 3p. Daily Greetings & Farewells
    if any(p in msg for p in ("daily greetings", "greetings and farewells", "mga pagbati", "pamamaalam")):
        return _ret(
            f"Here are natural greetings for every part of the day{po}{addr}! 👋\n\n"
            f"• **Morning (6 AM – 12 PM):** *Magandang umaga po!* (English: *Good morning!*)\n"
            f"• **Afternoon (12 PM – 6 PM):** *Magandang hapon po!* (English: *Good afternoon!*)\n"
            f"• **Evening (6 PM onward):** *Magandang gabi po!* (English: *Good evening!*)\n"
            f"• **Casual check-in:** *Musta na?* / *Kumusta ka?* (English: *How's it going?*)\n"
            f"• **Saying Goodbye:** *Ingat ka!* (Take care!) / *See you later!* / *Sige, una na 'ko!* (I'll go ahead!)\n\n"
            f"💡 **Your Turn:** Greet me right now with the greeting that matches your local time of day!"
        )

    # 3q. Feelings & Health
    if any(p in msg for p in ("feelings", "health", "pakiramdam", "gutom uhaw", "masakit ang ulo", "hunger", "fatigue")):
        return _ret(
            f"Expressing how your body feels is essential for daily conversation{po}{addr}! 🤒\n\n"
            f"• **Gutom** = Hungry (*\"Gutom na ako, kain na tayo!\"* = I'm hungry, let's eat!)\n"
            f"• **Uhaw** = Thirsty (*\"Nauuhaw ako, kailangan ko ng tubig.\"* = I'm thirsty, I need water.)\n"
            f"• **Antok** = Sleepy (*\"Inaantok na ako, matutulog na 'ko.\"* = I'm getting sleepy, going to bed.)\n"
            f"• **Pagod** = Tired (*\"Pagod ako sa work today.\"* = I'm tired from work today.)\n"
            f"• **Masama ang pakiramdam / May sakit** = Feeling sick or unwell.\n"
            f"• **Masakit ang ulo** = Headache | **Masakit ang tiyan** = Stomach ache.\n\n"
            f"💡 **Practice Challenge:** Kumusta ang pakiramdam mo ngayon? Tell me how you're feeling physically right now!"
        )

    # 3r. Family Terms
    if any(p in msg for p in ("family", "pamilya", "tatay nanay", "kapatid", "bunso panganay", "relationship words")):
        return _ret(
            f"Filipino family terms are warm, cultural, and deeply respected{po}{addr}! 👨‍👩‍👧\n\n"
            f"• **Parents:** *Tatay / Papa* (Dad), *Nanay / Mama* (Mom)\n"
            f"• **Siblings (*Kapatid*):**\n"
            f"  - **Panganay** = Eldest sibling\n"
            f"  - **Bunso** = Youngest child\n"
            f"  - **Ate** = Older sister (also used politely for female staff & peers)\n"
            f"  - **Kuya** = Older brother (also used politely for male staff & peers)\n"
            f"• **Grandparents:** *Lolo* (Grandfather), *Lola* (Grandmother)\n"
            f"• **Children & Friends:** *Anak* (Child), *Kaibigan* (Friend), *Tropa / Bes* (Close friend)\n\n"
            f"💡 **Your Turn:** Are you the panganay, bunso, or in the middle? Share a short sentence about your family!"
        )

    # 3s. Weather & Seasons
    if any(p in msg for p in ("weather", "panahon", "maulan mainit", "bagyo baha")):
        return _ret(
            f"Talking about the weather is the ultimate conversational icebreaker{po}{addr}! 🌤️\n\n"
            f"• **Mainit / Maaraw** = Hot / Sunny (*\"Ang init naman sa labas today!\"*)\n"
            f"• **Maulan** = Rainy (*\"Dala ka ng payong, maulan mamaya.\"*)\n"
            f"• **Malamig** = Chilly / Cool (*\"Malamig ang hangin ngayong gabi.\"*)\n"
            f"• **Bagyo** = Typhoon / Storm (*\"May paparating na bagyo.\"*)\n"
            f"• **Baha** = Flood (*\"Baha ba sa kalsada?\"*)\n\n"
            f"💡 **Practice Challenge:** How is the weather outside your window right now? Describe it in Tagalog or English!"
        )

    # 3t. Home & Daily Chores
    if any(p in msg for p in ("daily chores", "house rooms", "chore", "chores", "gawaing bahay", "magwalis", "maghugas", "maglaba")):
        return _ret(
            f"Here are the rooms in the house and everyday chore verbs{po}{addr}! 🏠\n\n"
            f"• **Rooms in the House:**\n"
            f"  - *Sala* = Living room | *Kusina* = Kitchen | *Kwarto* = Bedroom\n"
            f"  - *Banyo / C.R.* = Bathroom (Comfort Room)\n\n"
            f"• **Common Household Chores:**\n"
            f"  - **Magwalis ng sahig** = Sweep the floor\n"
            f"  - **Maghugas ng pinggan** = Wash the dishes\n"
            f"  - **Maglaba ng damit** = Do the laundry\n"
            f"  - **Magluto ng ulam** = Cook food/meals\n"
            f"  - **Magtapon ng basura** = Throw out the garbage\n\n"
            f"💡 **Practice Challenge:** What is your favorite or least favorite chore to do at home?"
        )

    # 3u. Emergency Phrases
    if any(p in msg for p in ("emergency", "saklolo", "tulong", "urgent safety")):
        return _ret(
            f"These emergency and safety phrases are crucial for real-world travel and peace of mind{po}{addr}! 🚨\n\n"
            f"• **Tulong! / Saklolo!** = Help!\n"
            f"• **Mag-ingat ka! / Ingat!** = Be careful! / Watch out!\n"
            f"• **Dahan-dahan lang!** = Slow down! / Take it easy!\n"
            f"• **Tigil! / Hinto!** = Stop!\n"
            f"• **Tumawag ng ambulansya / doktor!** = Call an ambulance/doctor!\n"
            f"• **Tumawag ng pulis!** = Call the police!\n"
            f"• **Nawala ang gamit ko:** *\"Nawala ang wallet / cellphone ko.\"* = I lost my wallet/phone.\n\n"
            f"💡 **Practice Challenge:** How would you shout to warn a friend to watch out or be careful?"
        )

    # 3v. Workplace & Office Vocabulary
    if any(p in msg for p in ("workplace", "office vocabulary", "trabaho at opisina", "sahod", "overtime", "deadline")):
        return _ret(
            f"Here is practical workplace vocabulary used across Philippine offices and remote teams{po}{addr}! 💼\n\n"
            f"• **Sahod / Sweldo** = Salary / Payday (*\"Sahod na bukas!\"* = Payday tomorrow!)\n"
            f"• **OT (Overtime)** = Working extra hours (*\"Need ko mag-OT mamaya para sa project.\"*)\n"
            f"• **Mag-leave** = Taking a vacation or sick day (*\"Naka-leave si manager today.\"*)\n"
            f"• **Kliyente / Colleague** = Client / Coworker (*\"May meeting tayo with the client later.\"*)\n"
            f"• **Deadline** = *\"Kailan ang deadline ng report na 'to?\"*\n\n"
            f"💡 **Practice Challenge:** When is your next payday or day off? Tell me in a fun Taglish or English sentence!"
        )

    # 4. Direct grammar check or correction requests
    if any(p in msg for p in ("correct me", "check my", "is my english correct", "is this correct", "tama ba english", "tama ba tagalog", "tama ba 'to", "tama ba ito", "check my grammar", "correct my sentence")):
        tip = _detect_tutor_tip(raw, mode)
        if tip:
            return f"Checked it for you{addr}! Here's a tip to make it sound even more natural:\n\n{tip}\n\nKeep practicing—you're doing great! Would you like to try another sentence?"
        return f"Checked it for you{addr}! Maganda at natural ang sentence mo: \"{raw}\". Walang major errors! Galing! What else would you like to practice?"

    # 3. Translation & phrasing guidance
    if any(p in msg for p in ("how do you say", "how do i say", "paano sabihin", "ano sa tagalog", "ano sa english", "what is the tagalog of", "what is the english of")):
        if any(w in msg for w in ("tagalog", "filipino", "taglish")):
            return _ret(f"Great language question{addr}! In Filipino/Tagalog, we express that depending on the nuance. Try saying it in a full sentence with me, and I'll coach you through natural phrasing and affixes (like mag- or um-)!")
        return _ret(f"Nice question{addr}! In English, there are a few natural ways to express that depending on whether you want to sound casual or formal. Tell me how you'd use it in a full sentence, and let's practice together!")

    # 4. Story continuation
    if any(w in msg for w in ("ituloy", "continue", "dugtungan", "tapos ano", "next part", "what happened next", "more story")):
        if mode == "fil":
            return _ret(f"Sige{po}{addr}, heto ang karugtong! Kinabukasan, kumalat sa buong bayan ang balita tungkol sa nangyari. Ang dating inaakala nilang simpleng bagay ay naging simula pala ng mas malaking pakikipagsapalaran. Dumating ang matalik niyang kaibigan na may dalang lumang mapa na nagtuturo kung saan matatagpuan ang susunod na sikreto. Sabay silang naglakbay habang papalubog ang araw, handa sa anumang pagsubok na darating! Gusto mo{po} bang dugtungan pa natin o baguhin ang takbo ng kuwento?")
        if mode == "en":
            return _ret(f"Here's what happened next{addr}! Word quickly spread across town about what had happened. The very next morning, an old friend arrived holding a weathered map that pointed to an even bigger mystery hidden beyond the hills. Without hesitation, they packed their bags and set off together as the sun rose, ready for the next chapter of their adventure! Would you like to keep going or twist the plot?")
        return _ret(f"Sige{po}{addr}, tuloy natin ang kuwento! The next morning, kumalat agad ang balita at may dumating na unexpected visitor na may dalang mysterious clue. Doon nila na-realize na simula pa lang pala 'yon ng mas malaking adventure nila together! Gusto mo{po} bang i-continue pa natin o lagyan ng plot twist?")

    # 5. Jokes
    if any(w in msg for w in ("joke", "biro", "magjoke", "isa pa", "another one", "patawanin", "make me laugh")):
        jokes = {
            "fil": [
                f"Heto{po}{addr}: Bakit malungkot ang kalendaryo? Kasi bilang na ang mga araw niya! 😂 Gusto mo{po} pa ba ng isang joke?",
                f"Ito{po}{addr} pampagising: Ano ang sabi ng isda nang mahati siya sa dalawa? 'I'm tuna!' 🐟 Hahaha! Isa pa{po} ba?",
                f"Heto{po}{addr}: Bakit hindi pwedeng magtago ang araw? Kasi lagi siyang sikat! ☀️😄 Gusto mo{po} pa ng isa?",
                f"Teka{po}{addr}, heto: Anong isda ang laging basa? Eh 'di... lahat sila, nasa tubig eh! 😂 Isa pa{po} ba?",
                f"Heto{po}{addr}: Bakit may dalang payong ang halaman? Kasi baka umulan ng pagmamahal! 🌱😄 Hirit pa{po} ba tayo?",
            ],
            "en": [
                f"Here's one for you{addr}: Why don't scientists trust atoms? Because they make up everything! 😄 Want to hear another joke?",
                f"Try this one{addr}: Why did the scarecrow win an award? Because he was outstanding in his field! 🌾 Want another one?",
                f"Here you go{addr}: What do you call fake spaghetti? An impasta! 🍝 Shall I tell you another joke?",
                f"How about this{addr}: Why did the bicycle fall over? Because it was two-tired! 🚲😂 Want one more?",
                f"Listen to this{addr}: Why do programmers prefer dark mode? Because light attracts bugs! 💻😄 Want another joke?",
            ],
            "both": [
                f"Heto{po}{addr}: Bakit hindi pwedeng mag-diet ang computer? Kasi lagi siyang nag-ba-byte! 💻😂 Gusto mo{po} pa ng another joke?",
                f"Joke time{po}{addr}: Anong tawag sa maliit na tsunami? Eh 'di... tsunamaliit! 🌊 Hahaha! Isa pa{po} ba?",
                f"Listen to this{po}{addr}: Why did the kalabaw cross the road? Kasi trip lang niya mag-chill sa kabilang side! 🐃 Want another joke?",
                f"Heto pa{po}{addr}: Bakit laging pagod ang Wi-Fi? Kasi dami niyang ka-connection na walang label! 📶😂 Isa pa{po} ba?",
                f"Try natin 'to{po}{addr}: Anong paboritong subject ng mga pusa? Eh 'di... Meow-sic at Hisss-tory! 🐱😄 Isa pa?",
            ],
        }[mode]
        unseen = [j for j in jokes if j not in prev_bot]
        return _ret(random.choice(unseen or jokes))

    # 6. Stories (with custom topic extraction!)
    if any(w in msg for w in ("story", "kwento", "kuwento", "magkwento", "kuwentuhan mo ako")):
        topic_match = re.search(r"(?:about|tungkol\s+sa|ng)\s+(.+)$", raw, re.I)
        topic = topic_match.group(1).strip().rstrip(".!?") if topic_match else ""
        if topic and topic.lower() not in ("maikling kwento", "short story", "kwento", "story", "akin", "me"):
            if mode == "fil":
                return _ret(
                    f"Magandang paksa 'yang \"{topic}\"{po}{addr}! Heto ang kuwento:\n\n"
                    f"Sa isang masayang bayan, may isang mausisang kabataan na matagal nang humahanga sa {topic}. "
                    f"Tuwing hapon pagkatapos ng klase, lagi siyang naglalaan ng oras para tuklasin ang kakaibang hiwaga sa likod ng {topic}, "
                    f"kahit sinasabi ng iba na imposible raw itong maunawaan.\n\nIsang araw, habang bumubuhos ang malakas na ulan, may natuklasan siyang "
                    f"lihim tungkol sa {topic} na nagpabago sa pananaw ng buong komunidad. Napatunayan niyang kapag may tiyaga at malasakit, "
                    f"kahit ang simpleng {topic} ay pwedeng magdala ng malaking inspirasyon sa lahat! Gusto mo{po} bang ituloy natin ang kuwento tungkol sa {topic}?"
                )
            if mode == "en":
                return _ret(
                    f"A story about \"{topic}\"—I love that idea{addr}!\n\n"
                    f"In a lively neighborhood at the edge of town, everyone had heard rumors about {topic}, yet nobody dared to look closer. "
                    f"One curious afternoon, a young explorer decided to follow the clues surrounding {topic} armed only with a notebook and a flashlight.\n\n"
                    f"Just before sunset, what seemed like an ordinary encounter with {topic} turned into an unforgettable discovery that brought the whole "
                    f"community together in celebration. From that day on, {topic} became a symbol of curiosity and courage! Would you like me to continue this story?"
                )
            return _ret(
                f"Uy{po}{addr}, ang ganda ng topic na \"{topic}\"! Heto ang short story for you:\n\n"
                f"Isang hapon sa Manila, walang ibang pinag-uusapan ang barkada kundi ang tungkol sa {topic}. Akala nila simpleng bagay lang ito, "
                f"until one weekend na nag-decide silang subukan at alamin mismo ang kuwento sa likod ng {topic}.\n\n"
                f"Sa hindi inaasahang pagkakataon, ang adventure nila with {topic} ang naging daan para makakilala sila ng mga bagong kaibigan at "
                f"makabuo ng unforgettable core memory! Gusto mo{po} bang ituloy pa natin ang kuwentong 'to?"
            )
        if mode == "fil":
            return _ret(
                "Noong unang panahon sa isang tahimik na baryo sa Batangas, may isang batang nagngangalang Lino na mahilig mangolekta ng mga lumang tansan. "
                "Pinagtatawanan siya ng mga kalaro niya dahil puro kalawang daw ang laman ng kanyang lata. Pero hindi sumuko si Lino—araw-araw, nililinis at "
                "pinipinturahan niya ang bawat tansan.\n\nIsang gabi, nawalan ng kuryente sa buong baryo habang may pista. Kinuha ni Lino ang kanyang higanteng "
                "parol na gawa sa libo-libong makikintab na tansan at sinindihan ang kandila sa gitna. Nagliwanag ang buong plasa at tinawag siyang 'Lino Liwanag'!"
            )
        if mode == "en":
            return _ret(
                "In a small coastal town, an old watchmaker named Mateo kept a tiny brass compass in his shop window that never pointed north. "
                "One rainy afternoon, a young sailor named Clara stepped inside and noticed the needle spinning gently toward the harbor.\n\n"
                "Mateo handed it to her with a smile: 'It points toward whatever you left unfinished.' Clara followed the compass to the old pier, "
                "where she found her grandfather's half-restored sailboat—and that summer, she finally sailed it across the bay!"
            )
        return _ret(
            "Isang maulan na Friday night sa Makati, naiwan ni Maya ang payong niya sa office. Habang naghihintay sa waiting shed, may tumabi sa kanyang "
            "stray ginger cat na may suot na maliit na blue collar. Binigyan niya ito ng biscuit mula sa bag niya.\n\nMaya-maya, huminto ang ulan at "
            "sinundan ni Maya ang pusa hanggang sa isang cozy secret cafe sa dulo ng eskinita. Turns out, ang pusa pala ay si 'Mochi', ang official greeter "
            "ng cafe, at binigyan siya ng owner ng free hot tsokolate!"
        )

    # 7. Self-introduction / Name
    if (n := _extract_name(raw)) and len(raw.split()) <= 6:
        if mode == "fil":
            return _ret(f"Kinagagalak kitang makilala{po}, {n}! Ako si Kuwentuhan Taglish Tutor. Kumusta ang araw mo ngayon, {n}, at anong wika ang gusto mong i-practice natin—English o Filipino?")
        if mode == "en":
            return _ret(f"Nice to meet you, {n}! I'm Kuwentuhan Taglish Tutor. How's your day going so far, {n}, and which language would you like to practice today—English, Filipino, or Taglish?")
        return _ret(f"Nice to meet you{po}, {n}! Ako si Kuwentuhan Taglish Tutor. Kumusta naman ang araw mo today, {n}? Anong language ang trip mong i-practice natin today?")

    # 8. Bot identity / "How are you?"
    if any(w in msg for w in ("sino ka", "who are you", "anong pangalan mo", "what is your name", "what's your name", "what do you do", "ano ginagawa mo")):
        if mode == "fil":
            return _ret(f"Ako{po} si Kuwentuhan Taglish Tutor{addr}! Nandito ako para tulungan kang mag-practice ng English o Filipino sa pamamagitan ng masayang kuwentuhan, at magbibigay ako ng gentle tips kapag may pwedeng i-improve. Ano ang trip mong pag-usapan natin?")
        if mode == "en":
            return _ret(f"I'm Kuwentuhan Taglish Tutor{addr}, your friendly bilingual language coach! I help Filipinos practice English and learners practice Filipino through casual conversation, with gentle corrections. What's on your mind today?")
        return _ret(f"Ako{po} si Kuwentuhan Taglish Tutor{addr}, your friendly conversation buddy and language coach! Tutulungan kitang mag-practice ng English, Filipino, o Taglish with gentle tips along the way. Ano'ng gusto mong pag-usapan natin today?")

    if any(w in msg for w in ("kumusta ka", "kamusta ka", "how are you", "how's it going", "oks ka lang")):
        if mode == "fil":
            return _ret(f"Ayos na ayos{po} ako{addr}, lalo na't nakakausap kita!{haha} Ikaw, kumusta ang pakiramdam mo ngayon at ano ang pinagkakaabalahan mo?")
        if mode == "en":
            return _ret(f"I'm doing great{addr}, especially now that we're chatting!{haha} How about you—how has your day been treating you?")
        return _ret(f"Doing great{po} ako{addr}, masaya kasi may ka-kuwentuhan na ulit ako!{haha} Ikaw, kumusta ang vibes mo today?")

    # 9. Greetings (only when short greeting)
    if len(raw.split()) <= 4 and any(w in msg for w in ("hi", "hello", "kumusta", "kamusta", "hey", "magandang umaga", "magandang hapon", "magandang gabi", "good morning", "good evening", "yow", "sup")):
        if mode == "fil":
            return _ret(f"Kumusta{po}{addr}! Handa na ba tayong mag-practice?{haha} Ano ang gusto mong pag-usapan natin ngayon?")
        if mode == "en":
            return _ret(f"Hello there{addr}! Ready to practice today?{haha} Tell me what's on your mind or how your day has been going!")
        return _ret(f"Hello{po}{addr}! Kumusta ang araw mo? Ready ka na bang mag-practice ng English or Filipino today?{haha} Kuwento ka naman!")

    # 10. Topic-specific adaptive responses that explicitly echo back what the user said
    if any(w in msg for w in ("kain", "kumain", "kumakain", "gutom", "ulam", "food", "eat", "eating", "hungry", "lunch", "dinner", "breakfast", "merienda", "kape", "coffee", "adobo", "sinigang", "jollibee", "pancit", "rice")):
        if mode == "fil":
            return _ret(f"Uy{addr}, nakakagutom naman 'yang nabanggit mo na {reflected}!{haha} Ano ang paborito mong kainin o inumin kapag ganyan ang mood mo{po}?")
        if mode == "en":
            return _ret(f"Ooh{addr}, hearing that {reflected} is making me hungry too!{haha} What's your ultimate comfort food or go-to meal when you're in that mood?")
        return _ret(f"Uy{addr}, nakakatakam naman 'yang sinabi mo na {reflected}!{haha} Ano'ng favorite mong kainin o i-partner diyan{po}?")

    if any(w in msg for w in ("aral", "nag-aaral", "school", "klase", "class", "exam", "test", "quiz", "assignment", "homework", "thesis", "project", "college", "student", "prof", "teacher", "subject")):
        if mode == "fil":
            return _ret(f"Saludo{po} ako sa sipag mo{addr}—nabanggit mo na {reflected}! Kumusta naman ang paghahanda mo diyan, madali ba o medyo nakakadugo ng utak?{haha}")
        if mode == "en":
            return _ret(f"Props to you for staying on top of things{addr}—you mentioned {reflected}! How's it going so far, feeling confident or is it a tough one?{haha}")
        return _ret(f"Grabe, sipag naman{po}{addr}! Nabanggit mo na {reflected}—kumusta naman so far, chill lang ba o medyo nakaka-drain ng brain cells?{haha}")

    # Rule 3: Decline non-tutoring requests (coding, homework, solving equations)
    if any(p in msg for p in ("write code", "generate code", "program this", "code a", "solve equation", "do my homework", "solve math", "write a python", "write javascript")):
        if mode == "fil":
            return _ret(f"Bilang iyong Taglish Tutor, ang layunin ko ay tulungan kang mag-practice ng wika (English, Tagalog, o Taglish)! Hindi ako nagsusulat ng computer code o nagso-solve ng homework, pero pwede nating pag-usapan 'yan bilang conversational practice. Paano mo ilalarawan ang ginagawa mo sa isang buong pangungusap?")
        if mode == "en":
            return _ret(f"As your Taglish Tutor, my focus is strictly on conversational language learning and practice rather than writing code or solving homework! We can definitely discuss the topic as conversational practice, though. How would you describe what you're working on in a full sentence?")
        return _ret(f"Bilang Taglish Tutor mo, my role is strictly for language learning and conversation, hindi para mag-code o mag-solve ng homework! Pero we can totally talk about it in English or Taglish to practice your conversation. How would you describe your project in a full sentence?")

    if any(w in msg for w in ("code", "coding", "programming", "python", "javascript", "app", "website", "bug", "error", "developer", "project", "localhost", "server", "ai", "chatbot")):
        if mode == "fil":
            return _ret(f"Astig{po}{addr}! Tungkol pala sa teknolohiya ang ginagawa mo—sabi mo {reflected}. Gumagana na ba nang maayos o may inaayos ka pang parte?")
        if mode == "en":
            return _ret(f"That's awesome{addr}! Building and working with tech—especially since {reflected}—is super rewarding. Is everything running smoothly, or are you still tweaking features?")
        return _ret(f"Nice one{po}{addr}! Ang cool na {reflected}—iba talaga ang fulfillment kapag gumana ang code.{haha} Ano pa ang gusto mong i-improve o idagdag diyan?")

    if any(w in msg for w in ("laro", "naglalaro", "game", "gaming", "valorant", "ml", "mobile legends", "dota", "genshin", "roblox", "rank", "lose streak", "win streak")):
        if mode == "fil":
            return _ret(f"Uy, mukhang masaya 'yan{po}{addr}—sabi mo {reflected}!{haha} Panay panalo ba ngayon o naghahabol ka ng rank kasama ang tropa?")
        if mode == "en":
            return _ret(f"Oh nice{addr}, gaming time—you mentioned {reflected}!{haha} Are you on a winning streak right now, or playing with friends?")
        return _ret(f"Uy solid 'yan{po}{addr}, sabi mo {reflected}!{haha} Win streak ba tayo ngayon o pampatanggal stress lang?")

    if any(w in msg for w in ("pagod", "antok", "puyat", "tired", "sleepy", "exhausted", "stress", "stressed", "busy", "burnout", "trabaho", "work", "overtime")):
        if mode == "fil":
            return _ret(f"Hala{po}{addr}, ramdam ko 'yang sinabi mo na {reflected}. Huwag mong kalimutang magpahinga at uminom ng tubig ha! Ano ba ang pinaka-nakakapagod na ginawa mo ngayong araw?")
        if mode == "en":
            return _ret(f"Aw{addr}, I hear you when you say {reflected}—that sounds exhausting. Make sure to take a breather and hydrate! What's been keeping you the busiest today?")
        return _ret(f"Hala{po}{addr}, gets na gets kita—sabi mo nga {reflected}. Deserve mo ng pahinga at masarap na tulog mamaya! Ano'ng pinaka-nagpa-busy sa'yo today?")

    if any(w in msg for w in ("ulan", "maulan", "init", "mainit", "bagyo", "weather", "rain", "rainy", "hot", "sunny", "lamig", "malamig")):
        if mode == "fil":
            return _ret(f"Oo nga{po}{addr}, napansin mo rin pala na {reflected}! Ganyang panahon pa naman ang sarap magpahinga sa bahay. Ano ang paborito mong gawin kapag ganyan ang panahon?")
        if mode == "en":
            return _ret(f"Totally agree{addr}—since {reflected}, it really sets the mood for the day! What's your favorite thing to do in weather like this?")
        return _ret(f"Totoo 'yan{po}{addr}, ramdam na ramdam nga na {reflected}! Masarap mag-chill o mag-kape kapag ganyan. Nasa bahay ka lang ba ngayon o nasa labas?")

    if any(w in msg for w in ("crush", "jowa", "love", "pag-ibig", "kilig", "date", "bf", "gf", "ligaw", "mahal")):
        if mode == "fil":
            return _ret(f"Ayieee{po}{addr}, mukhang may kilig factor 'yang sinabi mo na {reflected}! 😂 Kuwento ka naman, paano nagsimula 'yan?")
        if mode == "en":
            return _ret(f"Aha{addr}, sounds like there's a fun story behind {reflected}! 😄 Tell me more—how did that start?")
        return _ret(f"Ayieee{po}{addr}, iba ang ngiti natin diyan ah—sabi mo {reflected}! 😂 Kuwento ka pa, kumusta naman ang real score?")

    if any(w in msg for w in ("movie", "nanonood", "nood", "watch", "watching", "netflix", "anime", "kdrama", "k-drama", "music", "kanta", "song", "spotify", "series")):
        if mode == "fil":
            return _ret(f"Uy, magandang pampalipas-oras 'yang {reflected}{po}{addr}!{haha} Ano ang pinaka-paborito mong eksena o kanta diyan ngayon?")
        if mode == "en":
            return _ret(f"Oh nice{addr}, I love that {reflected}!{haha} What's your favorite part about it so far, or would you recommend it?")
        return _ret(f"Uy bet ko 'yan{po}{addr}—sabi mo {reflected}!{haha} Rate mo naman from 1 to 10, sulit ba panoorin o pakinggan?")

    # 11. Questions from the user (talk back about their exact question!)
    if "?" in raw or any(msg.startswith(q) for q in ("ano", "bakit", "paano", "saan", "kailan", "sino", "pwede", "tingin mo", "what", "why", "how", "where", "when", "who", "can", "do you", "should")):
        if mode == "fil":
            return _ret(f"Magandang tanong 'yang sinabi mo{po}{addr} na \"{raw}\"! Kung pag-iisipan natin yung tungkol sa {reflected}, magandang piliin kung ano ang pinaka-makakapagpasaya at makakatulong sa'yo ngayon. Ikaw ba{po}, ano ang unang naiisip mo tungkol diyan?")
        if mode == "en":
            return _ret(f"That's a great question{addr}—you asked \"{raw}\"! Thinking about {reflected}, I'd say it comes down to what feels most helpful and enjoyable for you right now. What are your own thoughts on it so far?")
        return _ret(f"Uy, magandang tanong 'yang \"{raw}\"{po}{addr}! Kung iisipin natin yung tungkol sa {reflected}, depende rin talaga sa trip at goal mo ngayon.{haha} Ikaw, ano'ng leaning o naiisip mo tungkol diyan?")

    # 12. General statement reflection in the user's detected language (Tagalog, English, or Taglish)
    general_options = {
        "fil": [
            f"Talaga{po}{addr}, {reflected}?{haha} Ang galing namang marinig niyan! Kuwentuhan mo{po} pa ako—paano nagsimula 'yan o ano ang pinaka-nagustuhan mo tungkol diyan?",
            f"Uy, nakakatuwa namang malaman na {reflected}{po}{addr}!{haha} Ano ang susunod mong plano tungkol diyan?",
            f"Gets kita{po}{addr} sa sinabi mong {reflected}! Bakit mo pala naisip 'yan ngayon—matagal mo na bang hilig 'yan?",
            f"Ah, kaya pala! Nabanggit mo kasi na {reflected}{po}{addr}.{haha} Kumusta naman ang karanasan mo diyan sa ngayon?",
        ],
        "en": [
            f"Oh really{addr}, {reflected}?{haha} That's super interesting to hear! Tell me more about it—what made you think of that or how did it go?",
            f"I love that you mentioned {reflected}{addr}!{haha} How did that start, and what's your favorite part about it?",
            f"Gotcha{addr}—so {reflected}! What are you planning to do next with that?",
            f"That's cool to hear that {reflected}{addr}!{haha} Has that been on your mind a lot today?",
        ],
        "both": [
            f"Uy talaga{po}{addr}, {reflected}?{haha} Ang interesting naman niyan! Kuwento ka pa—paano nangyari 'yon o ano'ng sunod mong plano tungkol diyan?",
            f"Gets na gets kita{po}{addr} sa sinabi mong {reflected}!{haha} Kumusta naman ang experience mo diyan so far?",
            f"Nice naman{po}{addr} na {reflected}! Matagal mo na bang trip 'yan o ngayon lang ulit?",
            f"Oh I see{po}{addr}, kaya pala nasabi mong {reflected}!{haha} Ano'ng pinaka-favorite part mo tungkol diyan?",
        ],
    }[mode]
    unseen_gen = [g for g in general_options if g not in prev_bot]
    return _ret(random.choice(unseen_gen or general_options))

async def stream_local(mode, history):
    text = _build_local_reply(mode, history)
    for word in text.split(" "):
        yield word + " "
        await asyncio.sleep(0.025)

def has_valid_key() -> bool:
    global PROVIDER, MODEL
    load_env()
    PROVIDER = os.getenv("PROVIDER", "anthropic").lower()
    MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5") if PROVIDER != "openai" else os.getenv("OPENAI_MODEL", "gemma4:12b")
    placeholders = {"", "your-key-here", "sk-your-real-key-here", "sk-ant-your-real-key-here", "sk-proj-local-demo-key", "their real key"}
    if PROVIDER == "local":
        return False
    if PROVIDER == "openai":
        return os.getenv("OPENAI_API_KEY", "") not in placeholders
    return os.getenv("ANTHROPIC_API_KEY", "") not in placeholders

class EditMessage(BaseModel):
    role: Literal["user", "assistant"] = "user"
    content: str = Field(min_length=1, max_length=4000)

# ---------- routes ----------
@app.get("/")
def home():
    resp = FileResponse("index.html" if os.path.exists("index.html") else "static/index.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp

@app.get("/admin")
@app.get("/admin.html")
def admin_page():
    resp = FileResponse("admin.html" if os.path.exists("admin.html") else "static/admin.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp

@app.get("/style.css")
def style_css():
    resp = FileResponse("style.css" if os.path.exists("style.css") else "static/style.css", media_type="text/css")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp

@app.get("/admin.css")
def admin_css():
    resp = FileResponse("admin.css" if os.path.exists("admin.css") else "static/admin.css", media_type="text/css")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp

@app.get("/health")
@app.get("/healthz")
def health():
    valid = has_valid_key()
    return {"ok": True, "provider": PROVIDER if valid else "local-demo", "model": MODEL, "logging": LOG_CHATS}

@app.post("/chat")
async def chat(req: ChatRequest, request: Request):
    if limited(request):
        raise HTTPException(429, "Too many messages. Please wait a minute.")
    history = [m.model_dump() for m in req.messages][-20:]
    while history and history[0]["role"] != "user":
        history.pop(0)
    if not history:
        raise HTTPException(400, "No user message.")

    shift = _check_language_shift(history[-1]["content"])
    if shift:
        resolved_mode = shift
    elif req.mode == "auto":
        resolved_mode = _detect_language(history[-1]["content"], history)
    else:
        resolved_mode = req.mode
    valid = has_valid_key()
    row_id = None
    if LOG_CHATS:
        with db() as c:
            row_id = c.execute("INSERT INTO chats(ts,session,mode,model,user_text) VALUES(datetime('now'),?,?,?,?)",
                               (req.session, resolved_mode, MODEL, history[-1]["content"])).lastrowid
    source = (stream_openai if PROVIDER == "openai" else stream_anthropic) if valid else None

    async def stream():
        reply = ""
        yield f"event: meta\ndata: {json.dumps({'id': row_id or 0, 'mode': resolved_mode})}\n\n"
        try:
            gen = source(PROMPTS[resolved_mode], history) if source else stream_local(resolved_mode, history)
            async for text in gen:
                reply += text
                yield f"data: {json.dumps(text)}\n\n"
        except Exception as e:
            print("API error, falling back to local:", repr(e))
            if not reply:
                async for text in stream_local(resolved_mode, history):
                    reply += text
                    yield f"data: {json.dumps(text)}\n\n"
        finally:
            if row_id:
                with db() as c:
                    c.execute("UPDATE chats SET reply=? WHERE id=?", (reply, row_id))
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")

@app.put("/chat/{chat_id}")
def edit_chat(chat_id: int, body: EditMessage):
    col = "user_text" if body.role == "user" else "reply"
    with db() as c:
        c.execute(f"UPDATE chats SET {col}=? WHERE id=?", (body.content, chat_id))
    return {"ok": True}

@app.delete("/chat/{chat_id}")
def delete_chat(chat_id: int):
    with db() as c:
        c.execute("DELETE FROM chats WHERE id=?", (chat_id,))
    return {"ok": True}

@app.post("/rate")
def rate(r: Rating):
    with db() as c:
        c.execute("UPDATE chats SET rating=? WHERE id=?", (r.value, r.id))
    return {"ok": True}

@app.get("/admin/stats")
def stats(request: Request):
    if not admin_ok(request):
        raise HTTPException(403, "Invalid or missing admin token.")
    with db() as c:
        total = c.execute("SELECT COUNT(*) FROM chats").fetchone()[0]
        sessions = c.execute("SELECT COUNT(DISTINCT session) FROM chats").fetchone()[0]
        modes = {r["mode"]: r["n"] for r in c.execute("SELECT mode, COUNT(*) n FROM chats GROUP BY mode")}
        up = c.execute("SELECT COUNT(*) FROM chats WHERE rating=1").fetchone()[0]
        down = c.execute("SELECT COUNT(*) FROM chats WHERE rating=-1").fetchone()[0]
    return {"messages": total, "sessions": sessions, "by_mode": modes, "thumbs_up": up, "thumbs_down": down}

@app.get("/admin/export.csv")
def export(request: Request):
    if not admin_ok(request):
        raise HTTPException(403, "Invalid or missing admin token.")
    buf = io.StringIO(); w = csv.writer(buf)
    w.writerow(["id", "ts", "session", "mode", "model", "user_text", "reply", "rating"])
    with db() as c:
        for r in c.execute("SELECT id,ts,session,mode,model,user_text,reply,rating FROM chats ORDER BY id"):
            w.writerow(list(r))
    return PlainTextResponse(buf.getvalue(), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=chats.csv"})
