import os
import re
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Any, Dict, List, Tuple

from rapidfuzz import fuzz

APP_DIR = os.path.dirname(__file__)

DATA_PATH_1 = os.path.join(os.path.dirname(APP_DIR), "data", "problems.json")
DATA_PATH_2 = os.path.join(APP_DIR, "data", "problems.json")
DATA_PATH = DATA_PATH_1 if os.path.exists(DATA_PATH_1) else DATA_PATH_2

app = FastAPI(title="Autism Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


# ===== Smart JSON cache =====
KB: List[Dict[str, Any]] = []
UI: Dict[str, Any] = {}
MTIME: float = 0.0


def norm_list(x: Any) -> List[str]:
    if not x:
        return []
    if isinstance(x, str):
        return [x.strip().lower()]
    if isinstance(x, list):
        out = []
        for v in x:
            if v is None:
                continue
            s = str(v).strip().lower()
            if s:
                out.append(s)
        return out
    s = str(x).strip().lower()
    return [s] if s else []


# ====== Universal keyword bank (bütün mövzular üçün) ======
KEYBANK: Dict[str, List[str]] = {
    "auditory": [
        "qulaq", "qulağ", "qulaqlar", "səs", "ses", "səs-küy", "ses-kuy",
        "tozsoran", "fen", "ventilyasiya", "ventilyator", "maşın səsi", "masin sesi",
        "qışqır", "bagir", "sirena", "metro", "toy"
    ],
    "smell": ["qoxu", "iy", "ətir", "parfum", "təmizlik", "xlor", "yemək qoxusu"],
    "touch": ["toxun", "paltar", "etiket", "corab", "corab", "düymə", "daraq", "saç"],
    "sleep": ["yuxu", "yatmır", "oyanır", "gecə", "rejim"],
    "toilet": ["tualet", "pampers", "bez", "unitaz", "karşok", "sidik", "nəcis"],
    "food": ["yemək", "qida", "dad", "tekstura", "seçir", "yemir", "qlüten", "kazein"],
    "meltdown": ["meltdown", "isterika", "qışqırır", "bağırır", "aqressiya", "vurur", "dişləyir"],
    "hygiene": ["diş", "fırça", "pasta", "duş", "çimmək", "dırnaq"],
    "school": ["məktəb", "bağça", "müəllim", "dərs", "inklyuziv"],
    "communication": ["danışmır", "ünsiyyət", "jest", "pecs", "exolaliya", "təkrar edir"],
}


def split_into_chunks(text: str) -> List[str]:
    """
    Uzun mətni 'başlıqlar' / boş sətirlər / bullet-lərə görə hissələrə bölür.
    Mətnin özünü dəyişmirik, sadəcə hissələrə ayırırıq.
    """
    if not text:
        return []

    # Normalizasiya
    t = text.replace("\r\n", "\n").strip()

    # 1) Başlıqlara görə böl ( **...** və ya sətirdə ':' olanlar )
    # 2) Sonra böyük boşluqlara görə böl
    parts = re.split(r"\n{2,}", t)
    chunks = []
    for p in parts:
        p = p.strip()
        if len(p) < 80:
            continue
        chunks.append(p)
    return chunks if chunks else [t]


def extract_keywords(title: str, text: str) -> List[str]:
    """
    Avtomatik keyword çıxarır:
    - title sözləri
    - KEYBANK-lə mövzu işarələri
    """
    title_l = (title or "").lower()
    text_l = (text or "").lower()

    kws = set()

    # title-dan sözlər
    for w in re.findall(r"[a-zəğıöşüç0-9]{3,}", title_l):
        kws.add(w)

    # keybank match
    for _, vocab in KEYBANK.items():
        for v in vocab:
            if v in title_l or v in text_l:
                kws.add(v)

    # praktik “ifadələr”
    if "qulaq" in text_l and ("ağlay" in text_l or "aglay" in text_l):
        kws.add("qulaqlarını tutub ağlayır")
        kws.add("qulaqlarini tutub aglayir")

    return sorted(kws)


def build_query_text(item: Dict[str, Any]) -> str:
    title = str(item.get("title", "")).lower()
    topic = str(item.get("topic", "")).lower()
    keywords = " ".join(norm_list(item.get("keywords")))
    text = str(item.get("text", "")).lower()   # 🔥 ƏN VACİB

    # çox uzun olmasın deyə ilk 800 simvol kifayətdir
    text = text[:800]

    return f"{title} {topic} {keywords} {text}".strip()


def keyword_overlap(user_text: str, keywords: List[str]) -> int:
    c = 0
    for kw in keywords:
        if kw and kw in user_text:
            c += 1
    return c


def top2_match(user_text: str) -> Tuple[Tuple[float, Optional[Dict[str, Any]]], Tuple[float, Optional[Dict[str, Any]]]]:
    scored: List[Tuple[float, Dict[str, Any], int]] = []

    for item in KB:
        kws = norm_list(item.get("keywords"))
        overlap = keyword_overlap(user_text, kws)

        q = build_query_text(item)
        if not q:
            continue

        s1 = fuzz.token_set_ratio(user_text, q)
        s2 = fuzz.partial_ratio(user_text, q)
        score = float(max(s1, s2))   # ✅ score burda yaranır

        # overlap bonus
        score += min(overlap * 4.0, 16.0)

        # overlap==0 üçün cəza (amma score hesablandıqdan sonra!)
        if overlap == 0:
            score -= 5.0

        scored.append((score, item, overlap))

    scored.sort(key=lambda x: x[0], reverse=True)

    top1 = (scored[0][0], scored[0][1]) if len(scored) > 0 else (0.0, None)
    top2 = (scored[1][0], scored[1][1]) if len(scored) > 1 else (0.0, None)
    return top1, top2

def load_json() -> None:
    global KB, UI, MTIME

    try:
        mtime = os.path.getmtime(DATA_PATH)
    except FileNotFoundError:
        KB = []
        UI = {}
        MTIME = 0.0
        return

    if mtime == MTIME:
        return

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    UI = data.get("ui", {}) or {}

    kb: List[Dict[str, Any]] = []

    # 1) items schema (qısa cavablar)
    for it in (data.get("items") or []):
        kb.append({
            "id": it.get("id"),
            "title": it.get("title", ""),
            "topic": it.get("topic", ""),
            "keywords": norm_list(it.get("keywords")),
            "text": it.get("text", ""),
            "source": "items",
        })

    # 2) problems schema (ekspert böyük mətnlər) → chunk-lara böl
    for p in (data.get("problems") or []):
        pid = p.get("id")
        title = p.get("title", "")
        desc = p.get("description", "")
        image = p.get("image")

        chunks = split_into_chunks(desc)
        for idx, ch in enumerate(chunks):
            kws = extract_keywords(title, ch)

            kb.append({
                "id": f"problem_{pid}_chunk_{idx}",
                "parent_id": pid,
                "title": title,
                "topic": "expert_article",
                "keywords": kws,
                "text": ch,              # ✅ mütəxəssis mətni eyni qalır (hissə-hissə)
                "source": "problems",
                "image": image,
            })

    KB = kb
    MTIME = mtime


@app.get("/config")
def config():
    load_json()
    return {
        "botName": UI.get("botName", "Autism Support Chatbot"),
        "subtitle": UI.get("subtitle", "Rəsmi dəstək • Praktik cavablar"),
        "welcomeMessage": UI.get(
            "welcomeMessage",
            "Salam. Suallarınızı yaza bilərsiniz — mən addım-addım cavablandıracağam."
        ),
        "inputPlaceholder": UI.get("inputPlaceholder", "Mesajınızı yazın..."),
        "buttonText": UI.get("buttonText", "Chat"),
    }


@app.get("/health")
def health():
    load_json()
    return {"ok": True, "data_path": DATA_PATH, "kb_items_count": len(KB)}


@app.post("/chat")
def chat(req: ChatRequest):
    load_json()
    user_text = (req.message or "").strip().lower()

    if not user_text:
        return {"answer": "Zəhmət olmasa sualınızı yazın.", "used_context": False, "sources": []}

    (s1, item1), (s2, _) = top2_match(user_text)

    # ✅ universal qərar qaydası (bütün suallar üçün)
    THRESHOLD = 80.0  # yüksək → lazımsız cavabları kəsir
    MARGIN = 4.0

    if item1 and s1 >= THRESHOLD and (s1 - s2) >= MARGIN:
        return {
            "answer": item1.get("text", "Cavab mövcud deyil."),
            "used_context": True,
            "sources": [{
                "id": item1.get("id"),
                "title": item1.get("title"),
                "source": item1.get("source"),
                "score": round(s1, 2),
                "parent_id": item1.get("parent_id"),
            }]
        }

    # Əmin deyilsə: lazımsız cavab yoxdur, yalnız dəqiqləşdirmə
    return {
        "answer": (
            "Sualınızı dəqiq tutmadım. Zəhmət olmasa 1-2 detal əlavə edin:\n"
            "• Yaş neçədir?\n"
            "• Nə tetikləyir? (səs, işıq, qoxu, toxunuş, rutin pozulması)\n"
            "• Nə qədər tez-tez olur?\n"
            "Məs: “5 yaş, tozsoran səsində qulaqlarını tutub ağlayır”."
        ),
        "used_context": False,
        "sources": []
    }
