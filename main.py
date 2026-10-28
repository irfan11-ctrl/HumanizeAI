import os
import re
import time
from collections import Counter
from typing import Any, Literal

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# =========================================================
# CONFIGURATION
# =========================================================

APP_NAME = "AI Humanizer"
APP_VERSION = "1.4.0"

# "ollama" untuk local | "groq" untuk production online
PROVIDER = os.getenv("PROVIDER", "groq").lower()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "180"))
MAX_INPUT_CHARACTERS = 12000
CHUNK_SIZE = 1800
CHUNK_OVERLAP = 120

ACTIVE_MODEL = GROQ_MODEL if PROVIDER == "groq" else OLLAMA_MODEL

# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title=APP_NAME,
    description="AI writing humanizer with local Ollama and cloud Groq support.",
    version=APP_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# SCHEMAS
# =========================================================

class HumanizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_INPUT_CHARACTERS)
    style: Literal["natural", "casual", "professional"] = "natural"
    mode: Literal["fast", "quality"] = "fast"


class HumanizeResponse(BaseModel):
    original_text: str
    humanized_text: str
    style: str
    mode: str
    model: str
    provider: str
    processing_time_seconds: float


class QualityAnalysisRequest(BaseModel):
    original_text: str = Field(..., min_length=1, max_length=MAX_INPUT_CHARACTERS)
    humanized_text: str = Field(..., min_length=1, max_length=MAX_INPUT_CHARACTERS)


# =========================================================
# STYLES & PROMPTS
# =========================================================

STYLE_INSTRUCTIONS = {
    "natural": (
        "Write like a clear, competent human. "
        "Sound natural and direct. Avoid anything that feels polished by AI."
    ),
    "casual": (
        "Write like a friendly, smart person explaining something in plain English. "
        "Keep it relaxed and conversational, but still clear."
    ),
    "professional": (
        "Write like a competent professional who values clarity. "
        "Stay polished but human — not corporate or stiff."
    ),
}


def build_prompt(text: str, style: str, pass_number: int = 1) -> str:
    style_instruction = STYLE_INSTRUCTIONS[style]

    if pass_number == 1:
        return f"""You are a skilled human editor. Rewrite the text so it sounds like it was written by a real person, not an AI.

STYLE:
{style_instruction}

CORE RULES:
- Keep the exact meaning, facts, names, numbers, and technical terms.
- Do not add new ideas, examples, or opinions.
- Do not remove important information.
- Change sentence structure when it improves flow.
- Mix short and longer sentences naturally.
- Prefer simple, concrete wording over fancy words.

STRICTLY AVOID these AI habits:
- Phrases like: "In today's world", "It is important to note", "Moreover", "Furthermore",
  "In conclusion", "As a result", "That being said", "In summary", "delve into",
  "landscape", "robust", "comprehensive", "leverage", "facilitate", "utilize"
- Overly balanced or perfectly parallel sentences
- Generic transitions and filler openings
- Sounding too smooth, too safe, or too polished

Return ONLY the rewritten text. No quotes, no labels, no explanation.

TEXT:
{text}

REWRITTEN:"""

    return f"""You are a human proofreader. Make this text sound more like something a real person would actually write.

STYLE:
{style_instruction}

GOALS:
- Keep the meaning identical.
- Make the wording feel more natural and less "generated".
- Break any remaining stiffness.
- Prefer how people normally speak or write in real life.
- You may slightly tighten or loosen sentences if it helps naturalness.
- Do not add or invent any new information.

AVOID:
- Anything that still feels robotic, overly formal, or template-like.
- Repeating the same sentence rhythm.

Return ONLY the final text. Nothing else.

TEXT:
{text}

FINAL:"""


def clean_output(text: str) -> str:
    text = text.strip()
    text = re.sub(
        r"^(REWRITTEN:|FINAL:|REWRITTEN TEXT:|FINAL TEXT:)\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"</?text>", "", text).strip()
    return text.strip("\"'")


# =========================================================
# PROVIDERS
# =========================================================

def generate_with_ollama(prompt: str, max_tokens: int = 1024) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "10m",
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.15,
            "num_predict": max_tokens,
            "num_ctx": 2048,
        },
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    text = data.get("response", "").strip()
    if not text:
        raise HTTPException(status_code=502, detail="Ollama returned an empty response.")
    return clean_output(text)


def generate_with_groq(prompt: str, max_tokens: int = 1024) -> str:
    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not configured on the server.",
        )

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a skilled human editor. Return only the rewritten text.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": max_tokens,
    }

    response = requests.post(
        GROQ_URL,
        headers=headers,
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()

    try:
        text = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError):
        raise HTTPException(status_code=502, detail="Groq returned an invalid response.")

    if not text:
        raise HTTPException(status_code=502, detail="Groq returned an empty response.")

    return clean_output(text)


def generate_text(prompt: str, max_tokens: int = 1024) -> str:
    try:
        if PROVIDER == "groq":
            return generate_with_groq(prompt, max_tokens=max_tokens)
        return generate_with_ollama(prompt, max_tokens=max_tokens)

    except HTTPException:
        raise
    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504,
            detail="The AI provider took too long to respond. Try a shorter text.",
        )
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to the AI provider. Check server configuration.",
        )
    except requests.exceptions.HTTPError as exc:
        status = getattr(exc.response, "status_code", 502)
        detail = f"AI provider HTTP error: {exc}"
        if status == 429:
            detail = "Rate limit reached. Please try again in a moment."
        raise HTTPException(status_code=502, detail=detail)
    except ValueError:
        raise HTTPException(status_code=502, detail="AI provider returned invalid JSON.")


# =========================================================
# HUMANIZE PIPELINE
# =========================================================

def split_into_chunks(text: str) -> list[str]:
    text = text.strip()
    if len(text) <= CHUNK_SIZE:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        if end >= len(text):
            chunks.append(text[start:].strip())
            break

        slice_ = text[start:end]
        last_period = max(slice_.rfind(". "), slice_.rfind("! "), slice_.rfind("? "))
        if last_period > CHUNK_SIZE * 0.5:
            end = start + last_period + 1

        chunks.append(text[start:end].strip())
        start = max(end - CHUNK_OVERLAP, start + 1)

    return [c for c in chunks if c]


def humanize_with_quality(text: str, style: str, mode: str = "fast") -> str:
    chunks = split_into_chunks(text)
    results = []

    for chunk in chunks:
        prompt1 = build_prompt(chunk, style, pass_number=1)
        max_tok = min(800, max(250, len(chunk) // 3 + 250))
        rewritten = generate_text(prompt1, max_tokens=max_tok)

        if mode == "quality" and len(rewritten.split()) >= 15:
            prompt2 = build_prompt(rewritten, style, pass_number=2)
            rewritten = generate_text(
                prompt2,
                max_tokens=min(800, max(200, len(rewritten) // 3 + 200)),
            )

        results.append(rewritten.strip())

    final = " ".join(results)
    final = re.sub(r"\s{2,}", " ", final).strip()
    return clean_output(final)


# =========================================================
# QUALITY ANALYSIS
# =========================================================

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then",
    "than", "so", "because", "as", "of", "at", "by",
    "for", "from", "in", "into", "on", "onto", "to",
    "with", "without", "about", "after", "before",
    "between", "during", "through", "over", "under",
    "is", "am", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did",
    "can", "could", "will", "would", "shall", "should",
    "may", "might", "must", "it", "its", "this", "that",
    "these", "those", "i", "me", "my", "mine", "we",
    "us", "our", "ours", "you", "your", "yours", "he",
    "him", "his", "she", "her", "hers", "they", "them",
    "their", "theirs", "who", "whom", "which", "what",
    "when", "where", "why", "how", "not", "no", "yes",
    "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "only", "very", "just",
    "also", "there", "here", "up", "down", "out", "now",
    "once", "again", "further", "own", "same", "too",
    "really", "still", "while", "although", "however",
    "therefore", "thus", "first", "second", "finally",
}


def count_syllables(word: str) -> int:
    word = word.lower().strip()
    if not word:
        return 0
    word = re.sub(r"[^a-z]", "", word)
    if not word:
        return 0
    if len(word) <= 3:
        return 1
    vowel_groups = re.findall(r"[aeiouy]+", word)
    count = len(vowel_groups)
    if word.endswith("e"):
        count -= 1
    if word.endswith("le") and len(word) > 2 and word[-3] not in "aeiouy":
        count += 1
    return max(1, count)


def split_sentences(text: str) -> list[str]:
    cleaned = text.strip()
    if not cleaned:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    return [s.strip() for s in sentences if s.strip()]


def analyze_text(text: str) -> dict[str, Any]:
    words = re.findall(r"\b[a-zA-Z]+(?:['’-][a-zA-Z]+)*\b", text)
    sentences = split_sentences(text)
    word_count = len(words)
    sentence_count = len(sentences)
    character_count = len(text)

    average_sentence_length = (
        round(word_count / sentence_count, 2) if sentence_count > 0 else 0.0
    )
    total_syllables = sum(count_syllables(w) for w in words)

    if word_count > 0 and sentence_count > 0:
        readability_score = (
            206.835
            - 1.015 * (word_count / sentence_count)
            - 84.6 * (total_syllables / word_count)
        )
        readability_score = round(max(0, min(100, readability_score)), 1)
    else:
        readability_score = None

    if readability_score is None:
        readability_label = "Not enough text"
    elif readability_score >= 80:
        readability_label = "Very easy"
    elif readability_score >= 60:
        readability_label = "Easy"
    elif readability_score >= 50:
        readability_label = "Fairly easy"
    elif readability_score >= 30:
        readability_label = "Difficult"
    else:
        readability_label = "Very difficult"

    normalized_words = [re.sub(r"[^a-z]", "", w.lower()) for w in words]
    content_words = [w for w in normalized_words if len(w) >= 4 and w not in STOP_WORDS]
    word_frequencies = Counter(content_words)

    repeated_words = [
        {"word": word, "count": count}
        for word, count in word_frequencies.most_common()
        if count >= 3
    ][:5]

    repeated_occurrences = sum(
        count - 1 for count in word_frequencies.values() if count >= 2
    )
    repetition_ratio = (
        round(repeated_occurrences / len(content_words) * 100, 1)
        if content_words
        else 0.0
    )

    return {
        "character_count": character_count,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "average_sentence_length": average_sentence_length,
        "readability_score": readability_score,
        "readability_label": readability_label,
        "repetition_ratio": repetition_ratio,
        "repeated_words": repeated_words,
    }


def compare_texts(
    original_text: str,
    humanized_text: str,
    original: dict[str, Any],
    humanized: dict[str, Any],
) -> dict[str, Any]:
    original_words = original["word_count"]
    humanized_words = humanized["word_count"]
    word_count_change = humanized_words - original_words
    word_count_change_percent = (
        round(word_count_change / original_words * 100, 1) if original_words > 0 else 0.0
    )

    original_chars = len(original_text)
    humanized_chars = len(humanized_text)
    character_count_change = humanized_chars - original_chars
    character_change_percent = (
        round(character_count_change / original_chars * 100, 1)
        if original_chars > 0
        else 0.0
    )

    original_readability = original["readability_score"]
    humanized_readability = humanized["readability_score"]
    readability_change = (
        round(humanized_readability - original_readability, 1)
        if original_readability is not None and humanized_readability is not None
        else None
    )

    return {
        "word_count_change": word_count_change,
        "word_count_change_percent": word_count_change_percent,
        "character_count_change": character_count_change,
        "character_change_percent": character_change_percent,
        "sentence_count_change": humanized["sentence_count"] - original["sentence_count"],
        "average_sentence_length_change": round(
            humanized["average_sentence_length"] - original["average_sentence_length"], 2
        ),
        "readability_score_change": readability_change,
        "repetition_ratio_change": round(
            humanized["repetition_ratio"] - original["repetition_ratio"], 1
        ),
    }


def create_analysis_summary(
    original: dict[str, Any],
    humanized: dict[str, Any],
    comparison: dict[str, Any],
) -> list[str]:
    summary = []

    word_change = comparison["word_count_change"]
    if word_change > 0:
        summary.append(f"The rewritten text contains {word_change} more words.")
    elif word_change < 0:
        summary.append(f"The rewritten text contains {abs(word_change)} fewer words.")
    else:
        summary.append("The word count remained the same.")

    sentence_change = comparison["sentence_count_change"]
    if sentence_change > 0:
        summary.append(f"The rewritten text has {sentence_change} more sentences.")
    elif sentence_change < 0:
        summary.append(f"The rewritten text has {abs(sentence_change)} fewer sentences.")
    else:
        summary.append("The sentence count remained the same.")

    if original["readability_score"] is not None and humanized["readability_score"] is not None:
        score_change = comparison["readability_score_change"]
        if score_change > 1:
            summary.append("The estimated readability score increased.")
        elif score_change < -1:
            summary.append("The estimated readability score decreased.")
        else:
            summary.append("The estimated readability score changed very little.")

    if humanized["repetition_ratio"] < original["repetition_ratio"]:
        summary.append("The repeated-word indicator decreased.")
    elif humanized["repetition_ratio"] > original["repetition_ratio"]:
        summary.append("The repeated-word indicator increased.")
    else:
        summary.append("The repeated-word indicator remained unchanged.")

    summary.append(
        "These are statistical estimates. They do not prove that "
        "the rewritten text preserves every original fact or meaning."
    )
    return summary


# =========================================================
# ROUTES
# =========================================================

@app.get("/")
def root():
    return {
        "app": APP_NAME,
        "status": "running",
        "version": APP_VERSION,
        "provider": PROVIDER,
        "model": ACTIVE_MODEL,
        "features": [
            "text_humanization",
            "writing_quality_analysis",
            "fast_mode",
            "quality_mode",
            "chunking",
        ],
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "app": APP_NAME,
        "provider": PROVIDER,
        "model": ACTIVE_MODEL,
    }


@app.post("/api/humanize", response_model=HumanizeResponse)
def humanize_text(request: HumanizeRequest):
    original_text = request.text.strip()
    if not original_text:
        raise HTTPException(status_code=400, detail="Please provide text to humanize.")

    start_time = time.perf_counter()
    humanized_text = humanize_with_quality(
        original_text,
        request.style,
        mode=request.mode,
    )
    processing_time = round(time.perf_counter() - start_time, 2)

    return HumanizeResponse(
        original_text=original_text,
        humanized_text=humanized_text,
        style=request.style,
        mode=request.mode,
        model=ACTIVE_MODEL,
        provider=PROVIDER,
        processing_time_seconds=processing_time,
    )


@app.post("/api/analyze-quality")
def analyze_writing_quality(request: QualityAnalysisRequest):
    original_text = request.original_text.strip()
    humanized_text = request.humanized_text.strip()

    if not original_text or not humanized_text:
        raise HTTPException(
            status_code=400,
            detail="Both original and rewritten text are required.",
        )

    original_analysis = analyze_text(original_text)
    humanized_analysis = analyze_text(humanized_text)
    comparison = compare_texts(
        original_text, humanized_text, original_analysis, humanized_analysis
    )
    summary = create_analysis_summary(original_analysis, humanized_analysis, comparison)

    return {
        "original": original_analysis,
        "humanized": humanized_analysis,
        "comparison": comparison,
        "summary": summary,
        "analysis_method": "local_statistical_analysis",
    }