"""Pure retrieval + answer-generation logic for palavbot.

Extracted from the original Streamlit app. No framework dependencies — safe to
import from FastAPI, a CLI, a Lambda handler, or tests.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import requests
from bs4 import BeautifulSoup
from openai import OpenAI

try:
    from youtube_transcript_api import YouTubeTranscriptApi
except BaseException:  # noqa: BLE001 - optional deps can raise pyo3 panics
    YouTubeTranscriptApi = None

try:
    import pdfplumber
except BaseException:  # noqa: BLE001 - optional deps can raise pyo3 panics
    pdfplumber = None


DEFAULT_LINKS_FILE = "palav_url_links.txt"
DEFAULT_INDEX_DIR = ".palav_index_cache"

CHUNK_CHARS = 3000
CHUNK_OVERLAP = 500
TOP_K = 10
MIN_SIM_THRESHOLD = 0.22

EMBED_MODEL = "text-embedding-3-small"
ANSWER_MODEL_DEFAULT = "gpt-4o-mini"


@dataclass
class DocChunk:
    id: str
    source_url: str
    title: str
    text: str


@dataclass
class IndexBundle:
    index: object
    vectors: np.ndarray
    chunks: List[DocChunk]
    report: Dict
    key: str
    paths: Dict[str, str]
    loaded_from_cache: bool


@dataclass
class AnswerResult:
    answer: str
    sources: List[Dict[str, str]]
    external_knowledge: bool
    rejected: bool


def normalize_whitespace(s: str) -> str:
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def file_sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_pdf_url(url: str) -> bool:
    return url.lower().split("?")[0].endswith(".pdf")


def is_youtube_url(url: str) -> bool:
    u = url.lower()
    return ("youtube.com/watch" in u) or ("youtu.be/" in u)


def extract_youtube_video_id(url: str) -> Optional[str]:
    if "youtu.be/" in url:
        return url.split("youtu.be/")[-1].split("?")[0].split("&")[0]
    m = re.search(r"[?&]v=([^&]+)", url)
    return m.group(1) if m else None


BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


def fetch_html_text(url: str, timeout: int = 20) -> Tuple[str, str]:
    r = requests.get(url, timeout=timeout, headers=BROWSER_HEADERS)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "aside"]):
        tag.decompose()
    title = soup.title.get_text(strip=True) if soup.title else url
    main = soup.find("main") or soup.find("article")
    text = main.get_text("\n", strip=True) if main else soup.get_text("\n", strip=True)
    return title, normalize_whitespace(text)


def fetch_pdf_text(url: str, timeout: int = 30) -> Tuple[str, str]:
    if pdfplumber is None:
        raise RuntimeError("pdfplumber is not installed.")
    r = requests.get(url, timeout=timeout, headers=BROWSER_HEADERS)
    r.raise_for_status()
    title = url
    pages_text: List[str] = []
    with pdfplumber.open(BytesIO(r.content)) as pdf:
        if pdf.metadata and "Title" in pdf.metadata:
            title = pdf.metadata["Title"]
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)
    return title, normalize_whitespace("\n".join(pages_text))


def fetch_youtube_transcript_text(url: str) -> Tuple[str, str]:
    if YouTubeTranscriptApi is None:
        raise RuntimeError("youtube-transcript-api is not installed.")
    vid = extract_youtube_video_id(url)
    if not vid:
        raise RuntimeError("Could not parse YouTube video id")

    # youtube-transcript-api v1.0 replaced the classmethod `get_transcript`
    # with an instance method `fetch` that returns a FetchedTranscript object.
    # Handle both so upgrades don't break the build.
    if hasattr(YouTubeTranscriptApi, "get_transcript"):
        try:
            raw = YouTubeTranscriptApi.get_transcript(vid, languages=["en"])
        except Exception:
            raw = YouTubeTranscriptApi.get_transcript(vid)
        segments = [x.get("text", "") for x in raw]
    else:
        api = YouTubeTranscriptApi()
        try:
            fetched = api.fetch(vid, languages=["en"])
        except Exception:
            fetched = api.fetch(vid)
        segments = [getattr(s, "text", "") for s in fetched]

    return f"YouTube transcript: {vid}", normalize_whitespace(" ".join(segments))


def chunk_text(text: str, chunk_chars: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_chars)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def embed_texts(client: OpenAI, texts: List[str]) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    faiss.normalize_L2(vecs)
    return vecs


def build_faiss_index(vectors: np.ndarray):
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


def load_allowed_urls(path: str) -> List[str]:
    """Extract URLs from the links file, one per occurrence.

    Lines can contain a descriptive prefix plus one or more http(s) URLs
    (e.g. ``"Title: https://a/b   https://c/d"``); all URLs are extracted
    and deduplicated. Lines with no URL (section headers, orphan
    descriptions) are skipped — the previous behavior of falling back to
    the raw line produced bogus entries the fetcher always failed on.
    """
    if not os.path.exists(path):
        return []
    urls: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            found = re.findall(r"https?://\S+", line)
            urls.extend(found)
    return list(dict.fromkeys(urls))


def index_key(links_file: str) -> str:
    if not os.path.exists(links_file):
        return "missing_links_file"
    h = file_sha1(links_file)
    settings = f"{EMBED_MODEL}|{CHUNK_CHARS}|{CHUNK_OVERLAP}"
    return sha1(h + "|" + settings)


def index_paths(key: str, index_dir: str) -> Dict[str, str]:
    return {
        "faiss": os.path.join(index_dir, f"{key}.faiss"),
        "vectors": os.path.join(index_dir, f"{key}.npy"),
        "chunks": os.path.join(index_dir, f"{key}.chunks.jsonl"),
        "report": os.path.join(index_dir, f"{key}.report.json"),
        "meta": os.path.join(index_dir, f"{key}.meta.json"),
    }


def index_exists(paths: Dict[str, str]) -> bool:
    return all(os.path.exists(paths[p]) for p in ["faiss", "vectors", "chunks", "meta"])


def save_index(paths: Dict[str, str], index, vectors: np.ndarray, chunks: List[DocChunk], report: Dict) -> None:
    faiss.write_index(index, paths["faiss"])
    np.save(paths["vectors"], vectors)
    with open(paths["chunks"], "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.__dict__, ensure_ascii=False) + "\n")
    with open(paths["report"], "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(paths["meta"], "w", encoding="utf-8") as f:
        json.dump(
            {
                "embed_model": EMBED_MODEL,
                "chunk_chars": CHUNK_CHARS,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            f,
            indent=2,
        )


def load_index(paths: Dict[str, str]) -> Tuple[object, np.ndarray, List[DocChunk], Dict]:
    index = faiss.read_index(paths["faiss"])
    vectors = np.load(paths["vectors"])
    chunks: List[DocChunk] = []
    with open(paths["chunks"], "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(DocChunk(**json.loads(line)))
    report: Dict = {}
    if os.path.exists(paths["report"]):
        with open(paths["report"], "r", encoding="utf-8") as f:
            report = json.load(f)
    return index, vectors, chunks, report


def ingest_sources(links_file: str) -> Tuple[List[DocChunk], Dict]:
    urls = load_allowed_urls(links_file)
    report: Dict = {"total_urls": len(urls), "ok": 0, "failed": []}
    chunks: List[DocChunk] = []
    for url in urls:
        try:
            if is_youtube_url(url):
                title, text = fetch_youtube_transcript_text(url)
            elif is_pdf_url(url):
                title, text = fetch_pdf_text(url)
            else:
                title, text = fetch_html_text(url)
            if len(text) < 200:
                raise RuntimeError("Text too short.")
            for i, piece in enumerate(chunk_text(text)):
                chunks.append(DocChunk(id=sha1(url + f"::{i}"), source_url=url, title=title, text=piece))
            report["ok"] += 1
        except Exception as e:
            report["failed"].append({"url": url, "error": repr(e)})
        time.sleep(0.1)
    return chunks, report


def build_or_load(
    links_file: str = DEFAULT_LINKS_FILE,
    api_key: Optional[str] = None,
    index_dir: str = DEFAULT_INDEX_DIR,
    force_rebuild: bool = False,
) -> IndexBundle:
    os.makedirs(index_dir, exist_ok=True)
    key = index_key(links_file)
    paths = index_paths(key, index_dir)

    if (not force_rebuild) and index_exists(paths):
        index, vectors, chunks, report = load_index(paths)
        return IndexBundle(index, vectors, chunks, report, key, paths, True)

    if not api_key:
        raise RuntimeError(
            "Index cache not found and OPENAI_API_KEY not provided; cannot build index at runtime."
        )

    chunks, report = ingest_sources(links_file)
    client = OpenAI(api_key=api_key)
    vectors = embed_texts(client, [c.text for c in chunks])
    index = build_faiss_index(vectors)
    save_index(paths, index, vectors, chunks, report)
    return IndexBundle(index, vectors, chunks, report, key, paths, False)


SYSTEM_INSTRUCTIONS = """You are a breastfeeding education chatbot for an NGO.

RULE 1: If the question is about BREASTFEEDING or MATERNAL HEALTH, you MUST provide an answer. 
- You are encouraged to respond in the language used by the user (e.g., Burmese, Spanish, etc.) by translating the relevant information.
- First, check the provided SOURCES for the answer. 
- If the SOURCES do not have the specific answer, use your general training data. 
- If you use general training data, you MUST start your response with "EXTERNAL_KNOWLEDGE:".

RULE 2: If the question is totally UNRELATED to breastfeeding (e.g., broken bones, car repair, history), reply: 
"I do not have required information. Please try different question"

Keep the tone parent-friendly and practical."""

FILTER_SUFFIX = (
    "\nAt the end of your response, if you used information from the provided SOURCES, "
    "provide a list of those specific URLs after the tag 'USED_URLS:'. "
    'Example: USED_URLS: ["url1", "url2"]'
)

FALLBACK_TEXT = "I do not have required information. Please try different question"


def retrieve(
    client: OpenAI,
    index,
    chunks: List[DocChunk],
    query: str,
    top_k: int = TOP_K,
) -> List[Tuple[float, DocChunk]]:
    qvec = embed_texts(client, [query])
    sims, idxs = index.search(qvec, top_k)
    results: List[Tuple[float, DocChunk]] = []
    for score, i in zip(sims[0], idxs[0]):
        if i != -1:
            results.append((float(score), chunks[i]))
    return results


def make_answer(
    client: OpenAI,
    model: str,
    question: str,
    retrieved: List[Tuple[float, DocChunk]],
    history: Optional[List[Dict[str, str]]] = None,
) -> AnswerResult:
    context_blocks = [f"URL: {ch.source_url}\nSNIPPET: {ch.text}" for _, ch in retrieved]

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS + FILTER_SUFFIX}
    ]
    if history:
        for m in history:
            if m.get("role") in ("user", "assistant") and m.get("content"):
                messages.append({"role": m["role"], "content": m["content"]})
    messages.append(
        {
            "role": "user",
            "content": f"QUESTION: {question}\n\nSOURCES:\n" + "\n".join(context_blocks),
        }
    )

    resp = client.chat.completions.create(model=model, messages=messages, temperature=0)
    full_content = resp.choices[0].message.content.strip()

    if FALLBACK_TEXT.lower() in full_content.lower():
        return AnswerResult(answer=FALLBACK_TEXT, sources=[], external_knowledge=False, rejected=True)

    if "USED_URLS:" in full_content:
        answer_part, url_part = full_content.split("USED_URLS:", 1)
        used_urls = re.findall(
            r"https?://\S+",
            url_part.replace('"', "").replace("[", "").replace("]", ""),
        )
    else:
        answer_part = full_content
        used_urls = []

    external_knowledge = "EXTERNAL_KNOWLEDGE:" in answer_part
    if external_knowledge:
        answer_part = answer_part.replace("EXTERNAL_KNOWLEDGE:", "").strip()

    used_urls = list(dict.fromkeys(u.rstrip(",").rstrip(".") for u in used_urls))
    url_to_title = {ch.source_url: ch.title for _, ch in retrieved}
    sources = [{"url": u, "title": url_to_title.get(u, u)} for u in used_urls]

    return AnswerResult(
        answer=answer_part.strip(),
        sources=sources,
        external_knowledge=external_knowledge,
        rejected=False,
    )
