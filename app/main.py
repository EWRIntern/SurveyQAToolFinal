from __future__ import annotations
import os
import re
import json
import shutil
import functools
import numpy as np
from threading import Lock
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from sqlalchemy import create_engine, Integer, String, Text, DateTime, ForeignKey, text as sa_text
from sqlalchemy.orm import DeclarativeBase, mapped_column, Mapped, sessionmaker, Session
from datetime import datetime

from docx import Document as DocxDocument
from odf.opendocument import load as odt_load
from odf import text as odt_text, teletype

APP_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(APP_ROOT, "data"))
UPLOAD_DIR = os.getenv("UPLOAD_DIR", os.path.join(DATA_DIR, "uploads"))

# Set DATABASE_URL to your Neon Postgres connection string in Render env vars.
# Falls back to SQLite for local development.
_sqlite_path = os.path.join(DATA_DIR, "app.db")
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{_sqlite_path}")
IS_POSTGRES = DATABASE_URL.startswith(("postgresql", "postgres"))

# OpenAI is only used for answer generation (chat), not embeddings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
CANONICAL_SIM_THRESHOLD = float(os.getenv("CANONICAL_SIM_THRESHOLD", "0.86"))
LOCAL_EMBED_MODEL = os.getenv("LOCAL_EMBED_MODEL", "all-MiniLM-L6-v2")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -------------------------
# DB
# -------------------------
class Base(DeclarativeBase):
    pass

class Document(Base):
    __tablename__ = "documents"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String(128), default="default")
    title: Mapped[str] = mapped_column(String(512))
    year: Mapped[int] = mapped_column(Integer)
    audience: Mapped[str] = mapped_column(String(32))  # member|voter
    filename: Mapped[str] = mapped_column(String(512))
    storage_path: Mapped[str] = mapped_column(String(1024))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

class Canonical(Base):
    __tablename__ = "canonical_questions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String(128), default="default")
    audience: Mapped[str] = mapped_column(String(32))  # member|voter (strict isolation)
    label: Mapped[str] = mapped_column(String(512))
    appearance_count: Mapped[int] = mapped_column(Integer, default=0)

class Question(Base):
    __tablename__ = "questions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_id: Mapped[int] = mapped_column(ForeignKey("documents.id"))
    tenant_id: Mapped[str] = mapped_column(String(128), default="default")
    canonical_id: Mapped[Optional[int]] = mapped_column(ForeignKey("canonical_questions.id"), nullable=True)

    question_code: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    section: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)

    question_text: Mapped[str] = mapped_column(Text)
    question_type: Mapped[str] = mapped_column(String(64))

    answer_options_json: Mapped[str] = mapped_column(Text, default="[]")
    matrix_rows_json: Mapped[str] = mapped_column(Text, default="[]")
    matrix_cols_json: Mapped[str] = mapped_column(Text, default="[]")

    topics_json: Mapped[str] = mapped_column(Text, default="[]")

    source_locator: Mapped[str] = mapped_column(String(512))
    source_quote: Mapped[str] = mapped_column(Text)

    audience: Mapped[str] = mapped_column(String(32))
    year: Mapped[int] = mapped_column(Integer)
    title: Mapped[str] = mapped_column(String(512))

class Embedding(Base):
    __tablename__ = "embeddings"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    question_id: Mapped[int] = mapped_column(ForeignKey("questions.id"))
    tenant_id: Mapped[str] = mapped_column(String(128), default="default")
    vector_json: Mapped[str] = mapped_column(Text)
    model: Mapped[str] = mapped_column(String(128))

if IS_POSTGRES:
    engine = create_engine(
        DATABASE_URL,
        pool_size=5,
        max_overflow=2,
        pool_timeout=30,
        pool_recycle=1800,
    )
else:
    os.makedirs(DATA_DIR, exist_ok=True)
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def setup_search_index():
    """Set up fast keyword search: GIN index on Postgres, FTS5 virtual table on SQLite."""
    with engine.connect() as conn:
        if IS_POSTGRES:
            conn.execute(sa_text("""
                CREATE INDEX IF NOT EXISTS questions_search_idx
                ON questions USING GIN(
                    to_tsvector('english',
                        COALESCE(question_text, '') || ' ' ||
                        COALESCE(title, '') || ' ' ||
                        COALESCE(answer_options_json, '') || ' ' ||
                        COALESCE(matrix_rows_json, '')
                    )
                )
            """))
        else:
            conn.execute(sa_text("""
                CREATE VIRTUAL TABLE IF NOT EXISTS questions_fts USING fts5(
                    question_id UNINDEXED,
                    tenant_id UNINDEXED,
                    search_text,
                    tokenize='unicode61'
                )
            """))
        conn.commit()

# -------------------------
# Utils
# -------------------------
def clean_ws(s: str) -> str:
    s = s.replace(" ", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def jdumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)

def jloads(s: str, default):
    try:
        return json.loads(s)
    except Exception:
        return default

def has_openai() -> bool:
    return bool(OPENAI_API_KEY)

# -------------------------
# Local embedding model (sentence-transformers, free, no API calls)
# -------------------------
_st_model = None
_st_lock = Lock()

def get_embed_model():
    global _st_model
    if _st_model is None:
        with _st_lock:
            if _st_model is None:
                from sentence_transformers import SentenceTransformer
                _st_model = SentenceTransformer(LOCAL_EMBED_MODEL)
    return _st_model

@functools.lru_cache(maxsize=512)
def _embed_query_cached(text: str) -> Tuple[float, ...]:
    """Cache query embeddings so repeated searches cost nothing."""
    model = get_embed_model()
    vec = model.encode([text], normalize_embeddings=True)[0]
    return tuple(float(x) for x in vec)

def embed_texts(texts: List[str]) -> List[List[float]]:
    model = get_embed_model()
    vecs = model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
    return [v.tolist() for v in vecs]

# -------------------------
# In-memory embedding cache (numpy matrix multiply for sub-millisecond search)
# -------------------------
_cache_lock = Lock()
_embed_matrix: Optional[np.ndarray] = None  # shape (N, dim), pre-normalized rows
_embed_qids: List[int] = []
_embed_tids: List[str] = []

def refresh_embed_cache():
    """Reload all embeddings from DB into a numpy matrix. Called at startup and after each ingest."""
    global _embed_matrix, _embed_qids, _embed_tids
    db = SessionLocal()
    try:
        rows = db.query(Embedding).all()
        if not rows:
            with _cache_lock:
                _embed_matrix = None
                _embed_qids = []
                _embed_tids = []
            return
        qids = [r.question_id for r in rows]
        tid_map = {
            q.id: q.tenant_id
            for q in db.query(Question).filter(Question.id.in_(qids)).all()
        }
        vecs = np.array([json.loads(r.vector_json) for r in rows], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.where(norms == 0, 1.0, norms)
        with _cache_lock:
            _embed_matrix = vecs
            _embed_qids = qids
            _embed_tids = [tid_map.get(qid, "default") for qid in qids]
    finally:
        db.close()

# -------------------------
# Extraction (DOCX + ODT)
# -------------------------
QUESTION_INDEX_RE = re.compile(r"^\s*(?P<code>\d+|[A-Z])\s*[\.)\:]\s+(?P<text>.+\S)\s*$")
MATRIX_ROW_INDEX_RE = re.compile(r"^\s*(?:\[\s*\])?\s*(?P<label>[a-z])\s*\.\s+")
OPTION_RE = re.compile(r"^\s*.+?(?:\t+|\s{2,})\d{1,4}\s*$")
ATTACHED_OPTION_RE = re.compile(r"^\s*.+?\D\d{1,2}\s*$")
MATRIX_ROW_RE = re.compile(r"^\s*(?:\[\s*\])?\s*(?P<label>[a-z])\.\s+(?P<rest>.+)$")
OPTION_LINE_RE = re.compile(r"^\s*(?P<text>.+?)\s*(?:\t+|\s{2,})\s*(?P<code>\d{1,4})\s*$")
ATTACHED_CODE_RE = re.compile(r"^\s*(?P<text>.*?\D)\s*(?P<code>\d{1,4})\s*$")
MATRIX_ROW_FALLBACK_RE = re.compile(r"^\s*(?P<text>.+?)(?P<digits>\d{4,})\s*$")
OPTION_PREFIX_RE = re.compile(r"^\s*(?P<code>\d+|[A-Za-z])[\)\.\-]\s+(?P<text>\S.+)$")

def docx_all_paragraph_text(path: str) -> List[str]:
    import zipfile
    import xml.etree.ElementTree as ET
    from collections import defaultdict

    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    num_to_abs = {}
    abs_lvls = defaultdict(dict)

    def parse_numbering(z: zipfile.ZipFile):
        if "word/numbering.xml" not in z.namelist():
            return
        root = ET.fromstring(z.read("word/numbering.xml"))
        for absn in root.findall(".//w:abstractNum", ns):
            abs_id = absn.get(f"{{{ns['w']}}}abstractNumId")
            if abs_id is None:
                continue
            for lvl in absn.findall("./w:lvl", ns):
                ilvl = lvl.get(f"{{{ns['w']}}}ilvl")
                if ilvl is None:
                    continue
                fmt_el = lvl.find("./w:numFmt", ns)
                txt_el = lvl.find("./w:lvlText", ns)
                numFmt = fmt_el.get(f"{{{ns['w']}}}val") if fmt_el is not None else None
                lvlText = txt_el.get(f"{{{ns['w']}}}val") if txt_el is not None else None
                abs_lvls[str(abs_id)][str(ilvl)] = {"numFmt": numFmt, "lvlText": lvlText}
        for num in root.findall(".//w:num", ns):
            numId = num.get(f"{{{ns['w']}}}numId")
            abs_el = num.find("./w:abstractNumId", ns)
            abs_id = abs_el.get(f"{{{ns['w']}}}val") if abs_el is not None else None
            if numId and abs_id:
                num_to_abs[str(numId)] = str(abs_id)

    def format_counter(n: int, numFmt: str | None) -> str:
        if not numFmt or numFmt == "decimal":
            return str(n)
        if numFmt in ("upperLetter", "upper-letter"):
            n0 = max(1, n); out = ""
            while n0 > 0:
                n0, rem = divmod(n0 - 1, 26)
                out = chr(65 + rem) + out
            return out
        if numFmt in ("lowerLetter", "lower-letter"):
            n0 = max(1, n); out = ""
            while n0 > 0:
                n0, rem = divmod(n0 - 1, 26)
                out = chr(97 + rem) + out
            return out
        return str(n)

    def apply_lvl_text(counter_str: str, lvlText: str | None) -> str:
        if not lvlText:
            return counter_str + "."
        return re.sub(r"%\d+", counter_str, lvlText)

    def paras_from_xml(xml_bytes: bytes, counters) -> List[str]:
        root = ET.fromstring(xml_bytes)
        out: List[str] = []
        for p in root.findall(".//w:p", ns):
            texts = [t.text for t in p.findall(".//w:t", ns) if t.text]
            s = clean_ws("".join(texts)) if texts else ""
            numPr = p.find("./w:pPr/w:numPr", ns)
            prefix = ""
            if numPr is not None:
                numId_el = numPr.find("./w:numId", ns)
                ilvl_el = numPr.find("./w:ilvl", ns)
                numId = numId_el.get(f"{{{ns['w']}}}val") if numId_el is not None else None
                ilvl = ilvl_el.get(f"{{{ns['w']}}}val") if ilvl_el is not None else None
                if numId is not None and ilvl is not None:
                    key = (str(numId), str(ilvl))
                    counters[key] += 1
                    abs_id = num_to_abs.get(str(numId))
                    fmt = lvlText = None
                    if abs_id is not None:
                        d = abs_lvls.get(abs_id, {}).get(str(ilvl), {})
                        fmt = d.get("numFmt")
                        lvlText = d.get("lvlText")
                    counter_str = format_counter(counters[key], fmt)
                    prefix = clean_ws(apply_lvl_text(counter_str, lvlText))
            if s:
                out.append((prefix + " " + s).strip() if prefix else s)
        return out

    paras: List[str] = []
    with zipfile.ZipFile(path, "r") as z:
        parse_numbering(z)
        counters = defaultdict(int)
        if "word/document.xml" in z.namelist():
            paras.extend(paras_from_xml(z.read("word/document.xml"), counters))
        for name in z.namelist():
            if name.startswith("word/header") or name.startswith("word/footer"):
                paras.extend(paras_from_xml(z.read(name), counters))

    dedup: List[str] = []
    for p in paras:
        if not dedup or dedup[-1] != p:
            dedup.append(p)
    return dedup


def extract_answer_options(lines: List[str]) -> List[str]:
    opts = []
    for ln in lines:
        ln = ln.strip()
        if re.match(r"^(\d+|[A-Za-z])[\)\.\-]\s+\S+", ln):
            opts.append(re.sub(r"^(\d+|[A-Za-z])[\)\.\-]\s+", "", ln).strip())
        elif re.match(r"^(Strongly|Somewhat|Neither|Very|Not sure|Don'?t know)\b", ln, re.IGNORECASE):
            opts.append(ln)
    seen = set()
    out = []
    for o in opts:
        k = o.lower()
        if k and k not in seen:
            seen.add(k)
            out.append(o)
    return out


@dataclass
class ExtractedQuestion:
    question_text: str
    question_type: str
    question_code: Optional[str]
    section: Optional[str]
    answer_options: List[str]
    matrix_rows: List[str]
    matrix_cols: List[str]
    source_locator: str
    source_quote: str

def docx_extract(path: str) -> List[ExtractedQuestion]:
    paras = docx_all_paragraph_text(path)

    def is_new_question_line(line: str) -> bool:
        line = clean_ws(line)
        if not line:
            return False
        if MATRIX_ROW_INDEX_RE.match(line):
            return False
        if OPTION_RE.match(line) or ATTACHED_OPTION_RE.match(line):
            return False
        return bool(QUESTION_INDEX_RE.match(line))

    questions: List[ExtractedQuestion] = []
    buf: List[str] = []
    buf_start = 0
    current_section = None

    def flush(end_idx: int):
        nonlocal buf, buf_start, current_section
        if not buf:
            return
        block_lines = [clean_ws(x) for x in buf if clean_ws(x)]
        buf = []
        if not block_lines:
            return
        first = block_lines[0]
        if first.isupper() and len(block_lines) == 1 and 6 <= len(first) <= 80:
            current_section = first
            return
        m = QUESTION_INDEX_RE.match(first)
        if not m:
            return
        qcode = m.group("code").strip()
        qtext = m.group("text").strip()
        rest = block_lines[1:]
        opts = extract_answer_options(rest)
        matrix_rows: List[str] = []
        for ln in rest:
            ln = clean_ws(ln)
            mm = MATRIX_ROW_RE.match(ln)
            if mm:
                matrix_rows.append(mm.group("rest").strip())
                continue
            mf = MATRIX_ROW_FALLBACK_RE.match(ln)
            if mf and len(mf.group("digits")) >= 4:
                matrix_rows.append(mf.group("text").strip())
        qtype = "open"
        if matrix_rows:
            qtype = "matrix"
        elif opts:
            qtype = "single"
        if any("select all" in l.lower() for l in block_lines):
            qtype = "multi"
        questions.append(ExtractedQuestion(
            question_text=qtext, question_type=qtype, question_code=qcode,
            section=current_section, answer_options=opts, matrix_rows=matrix_rows,
            matrix_cols=[], source_locator=f"docx:para:{buf_start}-{end_idx}",
            source_quote="\n".join(block_lines)[:900],
        ))

    for i, p in enumerate(paras):
        p = clean_ws(p)
        if not p:
            continue
        if is_new_question_line(p):
            if buf:
                flush(i - 1)
            buf_start = i
            buf = [p]
        else:
            if buf:
                buf.append(p)
    if buf:
        flush(len(paras) - 1)
    return questions


def odt_extract(path: str) -> List[ExtractedQuestion]:
    doc = odt_load(path)
    paras = doc.getElementsByType(odt_text.P)
    lines = [clean_ws(teletype.extractText(p)) for p in paras]
    lines = [l for l in lines if l]

    questions: List[ExtractedQuestion] = []
    buf: List[str] = []
    buf_start = 0
    current_section = None

    def flush(end_idx: int):
        nonlocal buf, buf_start, current_section
        if not buf:
            return
        block = "\n".join(buf).strip()
        buf = []
        parts = block.split("\n")
        first = parts[0].strip()
        if first.isupper() and len(parts) == 1 and 6 <= len(first) <= 80:
            current_section = first
            return
        m = QUESTION_INDEX_RE.match(first)
        qcode = m.group("code") if m else None
        rest = parts[1:]
        opts = extract_answer_options(rest)
        qtype = "open"
        if opts:
            qtype = "single"
        if any("select all" in l.lower() for l in parts):
            qtype = "multi"
        if re.search(r"\b0\s*to\s*10\b|\bscale\b|numeric entry|enter a number", block, re.IGNORECASE):
            qtype = "numeric"
        questions.append(ExtractedQuestion(
            question_text=first, question_type=qtype, question_code=qcode,
            section=current_section, answer_options=opts, matrix_rows=[],
            matrix_cols=[], source_locator=f"odt:line:{buf_start}-{end_idx}",
            source_quote=block[:700],
        ))

    for i, ln in enumerate(lines):
        is_new_q = bool(QUESTION_INDEX_RE.match(ln)) or (ln.endswith("?") and len(ln) > 15)
        if is_new_q and buf:
            flush(i - 1)
            buf_start = i
        buf.append(ln)
    flush(len(lines) - 1)
    return questions

def extract_questions(path: str, filename: str) -> List[ExtractedQuestion]:
    f = filename.lower()
    if f.endswith(".docx"):
        return docx_extract(path)
    if f.endswith(".odt"):
        return odt_extract(path)
    raise ValueError("Unsupported file type. Use .docx or .odt.")

# -------------------------
# Embeddings + Canonicals
# -------------------------
def question_to_embed_text(qtext: str, opts, rows, cols) -> str:
    parts = [qtext.strip()]
    if opts:
        parts.append("OPTIONS: " + " | ".join(opts[:60]))
    if rows:
        parts.append("ROWS: " + " | ".join(rows[:120]))
    if cols:
        parts.append("COLS: " + " | ".join(cols[:60]))
    return clean_ws("\n".join(parts))

def _cosine(a, b) -> float:
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def assign_to_canonical(db: Session, q: Question, vec: List[float]):
    canonicals = db.query(Canonical).filter(
        Canonical.tenant_id == q.tenant_id, Canonical.audience == q.audience
    ).all()
    best_cid = None
    best_sim = -1.0

    for c in canonicals:
        rep_q = (db.query(Question)
                 .filter(Question.canonical_id == c.id)
                 .order_by(Question.year.asc(), Question.id.asc())
                 .first())
        if not rep_q:
            continue
        rep_e = db.query(Embedding).filter(Embedding.question_id == rep_q.id).first()
        if not rep_e:
            continue
        rep_vec = jloads(rep_e.vector_json, [])
        if not rep_vec:
            continue
        sim = _cosine(vec, rep_vec)
        if sim > best_sim:
            best_sim = sim
            best_cid = c.id

    if best_cid is not None and best_sim >= CANONICAL_SIM_THRESHOLD:
        q.canonical_id = best_cid
        c = db.query(Canonical).filter(Canonical.id == best_cid).first()
        c.appearance_count += 1
        db.add_all([q, c])
        db.commit()
        return

    c = Canonical(tenant_id=q.tenant_id, audience=q.audience, label=q.question_text[:220], appearance_count=1)
    db.add(c); db.commit(); db.refresh(c)
    q.canonical_id = c.id
    db.add(q); db.commit()

# -------------------------
# FTS helpers
# -------------------------
def _fts_search_text(q: Question) -> str:
    return " ".join(filter(None, [
        q.question_text,
        q.title,
        " ".join(jloads(q.answer_options_json, [])),
        " ".join(jloads(q.matrix_rows_json, [])),
    ]))

def fts_insert_questions(db: Session, questions: List[Question]):
    if IS_POSTGRES:
        return  # GIN index on questions table is maintained automatically
    for q in questions:
        db.execute(sa_text(
            "INSERT INTO questions_fts(question_id, tenant_id, search_text) VALUES (:qid, :tid, :txt)"
        ), {"qid": q.id, "tid": q.tenant_id, "txt": _fts_search_text(q)})
    db.commit()

def fts_delete_questions(db: Session, question_ids: List[int]):
    if IS_POSTGRES:
        return  # GIN index on questions table is maintained automatically
    for qid in question_ids:
        db.execute(sa_text("DELETE FROM questions_fts WHERE question_id = :qid"), {"qid": qid})
    db.commit()

# -------------------------
# Ingest
# -------------------------
def ingest_document(db: Session, doc: Document) -> Dict[str, Any]:
    extracted = extract_questions(doc.storage_path, doc.filename)
    q_rows: List[Question] = []

    for ex in extracted:
        q = Question(
            doc_id=doc.id, tenant_id=doc.tenant_id, canonical_id=None,
            question_code=ex.question_code, section=ex.section,
            question_text=ex.question_text, question_type=ex.question_type,
            answer_options_json=jdumps(ex.answer_options),
            matrix_rows_json=jdumps(ex.matrix_rows),
            matrix_cols_json=jdumps(ex.matrix_cols),
            topics_json=jdumps([]),
            source_locator=ex.source_locator, source_quote=ex.source_quote,
            audience=doc.audience, year=doc.year, title=doc.title,
        )
        db.add(q)
        q_rows.append(q)

    db.commit()
    for q in q_rows:
        db.refresh(q)

    fts_insert_questions(db, q_rows)

    if q_rows:
        texts = [
            question_to_embed_text(
                q.question_text,
                jloads(q.answer_options_json, []),
                jloads(q.matrix_rows_json, []),
                jloads(q.matrix_cols_json, []),
            )
            for q in q_rows
        ]
        vectors = embed_texts(texts)
        for q, vec in zip(q_rows, vectors):
            e = Embedding(question_id=q.id, tenant_id=q.tenant_id,
                          vector_json=jdumps(vec), model=LOCAL_EMBED_MODEL)
            db.add(e)
        db.commit()

        for q, vec in zip(q_rows, vectors):
            assign_to_canonical(db, q, vec)

    refresh_embed_cache()

    return {
        "questions_extracted": len(q_rows),
        "embeddings_enabled": True,
        "embed_model": LOCAL_EMBED_MODEL,
        "canonical_threshold": CANONICAL_SIM_THRESHOLD,
    }

# -------------------------
# Search
# -------------------------
def serialize_question(q: Question) -> Dict[str, Any]:
    return {
        "question_id": q.id,
        "canonical_id": q.canonical_id,
        "survey_title": q.title,
        "year": q.year,
        "audience": q.audience,
        "section": q.section,
        "question_code": q.question_code,
        "question_type": q.question_type,
        "question_text": q.question_text,
        "answer_options": jloads(q.answer_options_json, []),
        "matrix_rows": jloads(q.matrix_rows_json, []),
        "matrix_cols": jloads(q.matrix_cols_json, []),
        "source_locator": q.source_locator,
        "source_quote": q.source_quote,
    }

def keyword_search(db: Session, tenant_id: str, query: str, limit: int = 80) -> List[Question]:
    if IS_POSTGRES:
        try:
            rows = db.execute(sa_text("""
                SELECT id FROM questions
                WHERE tenant_id = :tid
                AND to_tsvector('english',
                    COALESCE(question_text, '') || ' ' ||
                    COALESCE(title, '') || ' ' ||
                    COALESCE(answer_options_json, '') || ' ' ||
                    COALESCE(matrix_rows_json, '')
                ) @@ plainto_tsquery('english', :q)
                LIMIT :lim
            """), {"tid": tenant_id, "q": query, "lim": limit}).fetchall()
            qids = [r[0] for r in rows]
            if not qids:
                return []
            qs = db.query(Question).filter(Question.id.in_(qids)).all()
            by_id = {q.id: q for q in qs}
            return [by_id[qid] for qid in qids if qid in by_id]
        except Exception:
            pass  # fall through to LIKE
    else:
        try:
            safe_q = '"' + query.replace('"', '""') + '"'
            rows = db.execute(sa_text(
                "SELECT question_id FROM questions_fts WHERE tenant_id = :tid AND search_text MATCH :q ORDER BY rank LIMIT :lim"
            ), {"tid": tenant_id, "q": safe_q, "lim": limit}).fetchall()
            qids = [r[0] for r in rows]
            if not qids:
                return []
            qs = db.query(Question).filter(Question.id.in_(qids)).all()
            by_id = {q.id: q for q in qs}
            return [by_id[qid] for qid in qids if qid in by_id]
        except Exception:
            pass  # fall through to LIKE

    # LIKE fallback for both backends
    like = f"%{query}%"
    return (db.query(Question)
        .filter(Question.tenant_id == tenant_id)
        .filter(
            (Question.question_text.ilike(like)) |
            (Question.title.ilike(like)) |
            (Question.answer_options_json.ilike(like)) |
            (Question.matrix_rows_json.ilike(like))
        )
        .limit(limit).all()
    )

def semantic_search(tenant_id: str, query: str, limit: int = 80) -> List[int]:
    """Returns question_ids sorted by cosine similarity using in-memory numpy matrix."""
    with _cache_lock:
        mat = _embed_matrix
        qids = list(_embed_qids)
        tids = list(_embed_tids)

    if mat is None or not qids:
        return []

    qvec = np.array(_embed_query_cached(query), dtype=np.float32)
    mask = np.array([t == tenant_id for t in tids], dtype=bool)
    if not mask.any():
        return []

    sims = mat @ qvec
    sims = np.where(mask, sims, -2.0)
    top_idx = np.argsort(sims)[::-1][:limit]
    return [qids[i] for i in top_idx if sims[i] > -1.0]

def hybrid_search(db: Session, tenant_id: str, query: str, audience: Optional[str],
                  year_from: Optional[int], year_to: Optional[int], limit: int = 20):
    kw_qs = keyword_search(db, tenant_id, query, limit=120)
    sem_ids = semantic_search(tenant_id, query, limit=120)

    sem_qs: List[Question] = []
    if sem_ids:
        by_id = {q.id: q for q in db.query(Question).filter(Question.id.in_(sem_ids)).all()}
        sem_qs = [by_id[i] for i in sem_ids if i in by_id]

    scores: Dict[int, float] = {}
    for i, q in enumerate(kw_qs):
        scores[q.id] = scores.get(q.id, 0.0) + 1.0 / (1 + i)
    for i, q in enumerate(sem_qs):
        scores[q.id] = scores.get(q.id, 0.0) + 2.0 / (1 + i)

    def ok(q: Question) -> bool:
        if audience and q.audience != audience:
            return False
        if year_from is not None and q.year < year_from:
            return False
        if year_to is not None and q.year > year_to:
            return False
        return True

    all_qs = list({q.id: q for q in kw_qs + sem_qs}.values())
    cands = [q for q in all_qs if ok(q)]
    cands.sort(key=lambda q: scores.get(q.id, 0.0), reverse=True)
    return [serialize_question(q) for q in cands[:limit]]

def canonical_timeline(db: Session, canonical_id: int, audience: str):
    qs = (db.query(Question)
          .filter(Question.canonical_id == canonical_id, Question.audience == audience)
          .order_by(Question.year.asc(), Question.id.asc())
          .all())
    return [serialize_question(q) for q in qs]

def compare_years(db: Session, tenant_id: str, query: str, year_a: int, year_b: int,
                  audience: str, limit: int = 25):
    hits = hybrid_search(db, tenant_id, query, audience=audience,
                         year_from=None, year_to=None, limit=120)
    canon_ids = []
    seen = set()
    for h in hits:
        cid = h.get("canonical_id")
        if cid and cid not in seen:
            seen.add(cid); canon_ids.append(cid)

    out = []
    for cid in canon_ids:
        qa = db.query(Question).filter(Question.tenant_id == tenant_id, Question.audience == audience,
                                       Question.canonical_id == cid, Question.year == year_a).first()
        qb = db.query(Question).filter(Question.tenant_id == tenant_id, Question.audience == audience,
                                       Question.canonical_id == cid, Question.year == year_b).first()
        if qa or qb:
            out.append({
                "canonical_id": cid, "audience": audience, "year_a": year_a, "year_b": year_b,
                "question_a": serialize_question(qa) if qa else None,
                "question_b": serialize_question(qb) if qb else None,
            })
    out.sort(key=lambda r: (r["question_a"] is None, r["question_b"] is None))
    return out[:limit]

# -------------------------
# Generation (grounded, requires OpenAI key)
# -------------------------
def generate_answer(user_question: str, evidence: List[Dict[str, Any]],
                    audience: Optional[str], year_from: Optional[int],
                    year_to: Optional[int]) -> Dict[str, Any]:
    if not has_openai():
        raise RuntimeError("OPENAI_API_KEY not set; generation disabled.")
    from openai import OpenAI
    client = OpenAI()

    packed = [{
        "question_id": e["question_id"],
        "canonical_id": e.get("canonical_id"),
        "survey_title": e["survey_title"],
        "year": e["year"],
        "audience": e["audience"],
        "question_code": e.get("question_code"),
        "question_type": e.get("question_type"),
        "question_text": e["question_text"],
        "answer_options": e.get("answer_options", []),
        "matrix_rows": e.get("matrix_rows", []),
        "matrix_cols": e.get("matrix_cols", []),
        "source_locator": e.get("source_locator"),
    } for e in evidence[:18]]

    system = (
        "You are a survey questionnaire librarian.\n"
        "Use ONLY the EVIDENCE provided. Do not invent surveys, years, wording, options, or locations.\n"
        "Return VALID JSON with keys: answer (string), citations (array of evidence items you used), notes (optional string)."
    )
    payload = {
        "question": user_question,
        "filters": {"audience": audience, "year_from": year_from, "year_to": year_to},
        "EVIDENCE": packed,
    }
    resp = client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        temperature=0.2,
    )
    txt = resp.choices[0].message.content or ""
    try:
        data = json.loads(txt)
        if "answer" not in data:
            raise ValueError("missing answer")
        if "citations" not in data:
            data["citations"] = packed
        return data
    except Exception:
        return {"answer": txt.strip(), "citations": packed,
                "notes": "Model did not return valid JSON; returning raw output."}

# -------------------------
# FastAPI app
# -------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_search_index()
    refresh_embed_cache()
    yield

app = FastAPI(title="Survey Q&A Prototype", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(STATIC_DIR, "index.html"), "r", encoding="utf-8") as f:
        return f.read()

@app.get("/health")
def health():
    with _cache_lock:
        cached = len(_embed_qids)
    return {
        "ok": True,
        "embed_model": LOCAL_EMBED_MODEL,
        "cached_embeddings": cached,
        "generation": has_openai(),
    }

@app.post("/api/upload")
def api_upload(
    file: UploadFile = File(...),
    title: str = Form(...),
    year: int = Form(...),
    audience: str = Form(...),
    tenant_id: str = Form("default"),
    db: Session = Depends(get_db),
):
    audience = audience.strip().lower()
    if audience not in ("member", "voter"):
        raise HTTPException(400, "audience must be member or voter")
    if not (file.filename.lower().endswith(".docx") or file.filename.lower().endswith(".odt")):
        raise HTTPException(400, "Prototype supports .docx and .odt only")

    dest_path = os.path.join(UPLOAD_DIR, f"{tenant_id}__{year}__{file.filename}")
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    doc = Document(tenant_id=tenant_id, title=title, year=year, audience=audience,
                   filename=file.filename, storage_path=dest_path)
    db.add(doc); db.commit(); db.refresh(doc)

    summary = ingest_document(db, doc)

    # File is fully processed; delete it to avoid accumulating disk usage
    try:
        os.remove(dest_path)
    except Exception:
        pass

    return {"document_id": doc.id, "ingest_summary": summary}

@app.get("/api/documents")
def api_documents(tenant_id: str = "default", db: Session = Depends(get_db)):
    docs = (db.query(Document)
            .filter(Document.tenant_id == tenant_id)
            .order_by(Document.year.desc()).all())
    return [{"id": d.id, "title": d.title, "year": d.year,
             "audience": d.audience, "filename": d.filename} for d in docs]

@app.delete("/api/documents/{document_id}")
def api_delete_document(document_id: int, db: Session = Depends(get_db)):
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    questions = db.query(Question).filter(Question.doc_id == document_id).all()
    question_ids = [q.id for q in questions]
    canonical_ids = list({q.canonical_id for q in questions if q.canonical_id is not None})

    if question_ids:
        db.query(Embedding).filter(Embedding.question_id.in_(question_ids)).delete(synchronize_session=False)
        fts_delete_questions(db, question_ids)

    db.query(Question).filter(Question.doc_id == document_id).delete(synchronize_session=False)

    for canonical_id in canonical_ids:
        if not db.query(Question).filter(Question.canonical_id == canonical_id).first():
            db.query(Canonical).filter(Canonical.id == canonical_id).delete(synchronize_session=False)

    if document.storage_path and os.path.exists(document.storage_path):
        try:
            os.remove(document.storage_path)
        except Exception:
            pass

    db.delete(document)
    db.commit()

    refresh_embed_cache()

    return {"ok": True, "deleted_document_id": document_id}

@app.get("/api/search")
def api_search(
    q: str,
    tenant_id: str = "default",
    audience: Optional[str] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    limit: int = 20,
    db: Session = Depends(get_db),
):
    if audience:
        audience = audience.lower()
        if audience not in ("member", "voter"):
            raise HTTPException(400, "audience must be member or voter")
    return hybrid_search(db, tenant_id, q, audience, year_from, year_to, limit)

@app.get("/api/question/{question_id}")
def api_question(question_id: int, db: Session = Depends(get_db)):
    q = db.query(Question).filter(Question.id == question_id).first()
    if not q:
        raise HTTPException(404, "question not found")
    data = serialize_question(q)
    if q.canonical_id:
        data["timeline"] = canonical_timeline(db, q.canonical_id, audience=q.audience)
    return data

@app.get("/api/compare")
def api_compare(
    q: str,
    year_a: int,
    year_b: int,
    tenant_id: str = "default",
    audience: Optional[str] = None,
    limit: int = 25,
    db: Session = Depends(get_db),
):
    if audience:
        audience = audience.lower()
        if audience not in ("member", "voter"):
            raise HTTPException(400, "audience must be member or voter")
        return compare_years(db, tenant_id, q, year_a, year_b, audience, limit)
    return {
        "member": compare_years(db, tenant_id, q, year_a, year_b, "member", limit),
        "voter": compare_years(db, tenant_id, q, year_a, year_b, "voter", limit),
    }

@app.get("/api/answer")
def api_answer(
    q: str,
    tenant_id: str = "default",
    audience: Optional[str] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    k: int = 10,
    db: Session = Depends(get_db),
):
    if audience:
        audience = audience.lower()
        if audience not in ("member", "voter"):
            raise HTTPException(400, "audience must be member or voter")
    evidence = hybrid_search(db, tenant_id, q, audience, year_from, year_to, limit=k)
    try:
        gen = generate_answer(q, evidence, audience, year_from, year_to)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    return {
        "query": q, "evidence_count": len(evidence),
        "answer": gen.get("answer"), "citations": gen.get("citations", []),
        "notes": gen.get("notes"),
    }
