import re
from typing import Optional

import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from scipy.sparse import issparse

# ── Labels ────────────────────────────────────────────
LABELS = ["ham", "spam", "phishing"]
TAG_MAP = {"ham": ["Likely safe"], "spam": ["Spam"], "phishing": ["Phishing"]}

# ── Load model + pre-extract coefficients ─────────────
model = joblib.load("spam_model_v2.pkl")

_coefs = [cc.estimator.coef_ for cc in
          model.named_steps["clf"].calibrated_classifiers_]
AVG_COEF = np.mean(_coefs, axis=0)          # (3, n_features)
FEATURE_NAMES = model.named_steps["tfidf"].get_feature_names_out()

# Synthetic tokens injected by TextPreprocessor — useful for the model
# but meaningless as user-facing "flagged phrases", so we skip them.
SYNTHETIC_TOKENS = frozenset({
    "urltoken", "emailtoken", "capstoken", "moneytoken",
    "multiexclaim", "multiquestion", "numtoken",
})

# Common function words that are never meaningful security signals,
# regardless of their TF-IDF contribution.  An n-gram is filtered
# only when ALL of its tokens are in this set (so "click here" passes
# because "click" is not a stopword, while "here are" is filtered).
_FLAGGED_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "for", "in", "on", "at",
    "to", "of", "by", "as", "if", "so", "no", "up",
    "i", "me", "my", "we", "us", "our", "you", "your",
    "he", "she", "it", "its", "they", "them", "their",
    "this", "that", "these", "those", "who", "what", "which",
    "where", "when", "how", "than", "then",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "do", "does", "did",
    "will", "would", "can", "could", "may", "might", "shall", "should",
    "get", "got", "let", "make", "made", "take", "come", "go",
    "see", "know", "think", "say", "said", "tell", "give",
    "not", "very", "just", "also", "still", "already", "even",
    "more", "most", "much", "many", "some", "any", "all", "each",
    "here", "there", "now", "well", "too", "really", "about",
    "only", "over", "after", "before", "back", "out", "into",
    "one", "two", "new", "good", "great", "way", "day", "time",
    "thing", "things", "like", "look", "need", "want", "keep",
    "thats", "heres", "youre", "dont", "cant", "wont", "isnt",
    "whats", "weve", "youve", "theyre", "didnt", "doesnt",
})


# ── Request schema ────────────────────────────────────
class Message(BaseModel):
    text: str
    sender_domain: Optional[str] = None


# ── App ───────────────────────────────────────────────
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════
# SIGNAL ANALYSIS
# ═══════════════════════════════════════════════════════

# ── Flagged phrases: real TF-IDF feature extraction ───

def extract_flagged_phrases(text: str, predicted_class: int,
                            top_n: int = 6) -> list[str]:
    """Return the top-N input phrases whose TF-IDF × SVM-coefficient
    contribution was highest for the predicted class.

    Filters applied:
    - Ham predictions return [] (nothing to flag in safe email)
    - Minimum contribution threshold: 75th percentile of positive
      contributions AND an absolute floor of 0.01, so only features
      with genuinely strong signal qualify
    - Synthetic preprocessor tokens (urltoken etc.) excluded
    - N-grams composed entirely of stopwords excluded
    - If nothing survives filtering, returns [] rather than forcing
      weak features through
    """
    if predicted_class == 0:
        return []

    preprocessed = model.named_steps["preprocessor"].transform([text])
    tfidf_vec = model.named_steps["tfidf"].transform(preprocessed)

    tfidf_arr = (tfidf_vec.toarray()[0] if issparse(tfidf_vec)
                 else tfidf_vec[0])

    contributions = tfidf_arr * AVG_COEF[predicted_class]

    pos_mask = contributions > 0
    if not pos_mask.any():
        return []

    pos_values = contributions[pos_mask]
    threshold = max(np.percentile(pos_values, 75), 0.01)

    strong_idx = np.where(contributions >= threshold)[0]
    if len(strong_idx) == 0:
        return []

    ranked = strong_idx[np.argsort(contributions[strong_idx])[::-1]]

    phrases = []
    for idx in ranked:
        fname = FEATURE_NAMES[idx]
        tokens = fname.split()
        if any(t in SYNTHETIC_TOKENS for t in tokens):
            continue
        if all(t in _FLAGGED_STOPWORDS for t in tokens):
            continue
        if all(len(t) <= 1 for t in tokens):
            continue
        phrases.append(fname)
        if len(phrases) >= top_n:
            break

    return phrases


# ── Suspicious links: real URL/regex analysis ─────────

_URL_RE = re.compile(
    r'https?://\S+'
    r'|www\.\S+'
    r'|\b\S+\.(?:com|net|org|co\.uk|info|biz|win|club|xyz|tv|io|co)\b',
    re.IGNORECASE,
)

_IP_URL_RE = re.compile(r'https?://\d{1,3}(?:\.\d{1,3}){3}')

_SHORTENERS = frozenset({
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd",
    "buff.ly", "rebrand.ly", "shorturl.at", "tiny.cc", "cutt.ly",
})

_SUSPICIOUS_TLDS = re.compile(
    r'\.(?:ru|cn|xyz|win|club|biz|top|buzz|tk|ml|ga|cf)\b', re.IGNORECASE)

# Domains that embed a well-known brand name with extra words/hyphens
# (e.g. paypal-secure-verify.com, apple-id-support.net)
_BRAND_MIMIC_RE = re.compile(
    r'(?:paypal|apple|google|microsoft|amazon|netflix|bank|secure|verify'
    r'|account|login|support)[\w-]*\.'
    r'(?:com|net|org|info|co|xyz|biz|win|club)\b',
    re.IGNORECASE,
)


def analyze_links(text: str) -> str:
    """Score link suspicion based on real URL parsing.

    Checks:
      - Raw IP-based URLs                          (+3)
      - Known URL-shortener domains                (+2 each)
      - Non-HTTPS links (http://)                  (+1 each)
      - Suspicious TLDs (.ru, .xyz, .win, etc.)    (+2 each)
      - Domain mimics a known brand (e.g.
        paypal-secure-verify.com)                  (+2)
      - Excessive link count (>3)                  (+2)

    Returns "High" (>=4), "Medium" (>=2), or "Low".
    """
    urls = _URL_RE.findall(text)
    if not urls:
        return "Low"

    score = 0

    if _IP_URL_RE.search(text):
        score += 3

    for url in urls:
        low = url.lower()
        if any(s in low for s in _SHORTENERS):
            score += 2
        if low.startswith("http://"):
            score += 1
        if _SUSPICIOUS_TLDS.search(low):
            score += 2
        if _BRAND_MIMIC_RE.search(low):
            score += 2

    if len(urls) > 3:
        score += 2

    if score >= 4:
        return "High"
    if score >= 2:
        return "Medium"
    return "Low"


# ── Urgent language & financial incentive ─────────────
# Heuristic display layer: these curated phrase lists are used ONLY for
# the UI signal indicators.  They are NOT derived from the model's
# internal feature weights — the model makes its prediction independently
# via TF-IDF + SVM.  These lists simply highlight common threat patterns
# so the user gets a quick visual summary alongside the model's verdict.

_URGENCY_PHRASES = [
    "urgent", "immediately", "act now", "right away", "asap",
    "expires", "limited time", "last chance", "don't delay",
    "within 24 hours", "within 48 hours", "account will be",
    "suspended", "terminated", "closed", "locked", "verify now",
    "confirm now", "respond immediately", "action required",
    "your account has been", "unauthorized", "security alert",
]

# Hard financial indicators: scam-specific patterns rarely seen in
# legitimate email.  These trigger on their own.
_FINANCIAL_HARD = [
    "winner", "won", "lottery", "jackpot", "prize", "reward",
    "gift card", "congratulations", "claim your", "collect your",
    "no cost", "buy now",
]

# Soft financial indicators: words that appear in both scam AND
# legitimate contexts (newsletters, course platforms, charities).
# These only count when corroborated by urgency language or
# suspicious links — a freeCodeCamp email mentioning "free courses"
# shouldn't trigger financial_incentive on its own.
_FINANCIAL_SOFT = [
    "free", "offer", "deal", "discount", "cheap",
    "cash", "credit", "loan", "investment",
    "income", "earn", "profit",
]


def _score_urgency(text: str) -> str:
    lower = text.lower()
    hits = sum(1 for p in _URGENCY_PHRASES if p in lower)
    if hits >= 3:
        return "High"
    if hits >= 1:
        return "Medium"
    return "Low"


def _score_financial(text: str, urgency: str, links: str) -> str:
    """Score financial incentive with co-occurrence gating.

    Hard indicators (lottery, prize, winner, etc.) always count.
    Soft indicators (free, offer, earn, etc.) only count when
    corroborated by a STRONG risk signal:
      - Any urgency language (Medium or High), OR
      - High suspicious links (not Medium — a newsletter with many
        legitimate links scores Medium, which is not a scam signal)
    """
    lower = text.lower()

    hard = sum(1 for p in _FINANCIAL_HARD if p in lower)
    soft = sum(1 for p in _FINANCIAL_SOFT if p in lower)

    score = hard

    corroborated = urgency in ("Medium", "High") or links == "High"
    if corroborated:
        score += soft

    if score >= 3:
        return "High"
    if score >= 1:
        return "Medium"
    return "Low"


# ── Sender trust ──────────────────────────────────────

_TRUSTED_DOMAINS = frozenset({
    "google.com", "gmail.com", "microsoft.com", "outlook.com",
    "apple.com", "amazon.com", "linkedin.com", "github.com",
    "slack.com", "zoom.us", "salesforce.com",
})

_FREE_EMAIL_DOMAINS = frozenset({
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com",
    "icloud.com", "mail.com", "protonmail.com", "proton.me", "yandex.com",
    "gmx.com", "zoho.com",
})


def assess_sender(domain: Optional[str]) -> str:
    if not domain:
        return "Not available"
    d = domain.lower().strip()
    if d in _TRUSTED_DOMAINS:
        return "High"
    if d in _FREE_EMAIL_DOMAINS:
        return "Medium"
    return "Low"


# ═══════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════

@app.get("/")
def root():
    return FileResponse("spam-classifier-frontend/index.html")


@app.post("/predict")
def predict(msg: Message):
    probs = model.predict_proba([msg.text])[0]
    predicted_class = int(np.argmax(probs))
    prediction = LABELS[predicted_class]
    confidence = round(float(probs[predicted_class]), 2)

    # risk_score = 100 – (ham_probability × 100), clamped to [0, 100].
    # Rationale: ham probability is the model's belief the message is
    # safe.  Subtracting from 100 converts it to a threat score where
    # 0 = certainly safe and 100 = certainly dangerous.
    risk_score = max(0, min(100, round(100 - float(probs[0]) * 100)))

    # Compute independent signals first, then financial (needs the others)
    urgency = _score_urgency(msg.text)
    links = analyze_links(msg.text)
    financial = _score_financial(msg.text, urgency, links)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "risk_score": risk_score,
        "signals": {
            "urgent_language": urgency,
            "suspicious_links": links,
            "financial_incentive": financial,
            "sender_trust": assess_sender(msg.sender_domain),
        },
        "flagged_phrases": extract_flagged_phrases(msg.text,
                                                   predicted_class),
        "category_tags": TAG_MAP[prediction],
    }
