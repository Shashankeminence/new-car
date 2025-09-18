import os
import json
import math
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "faiss-cpu is required for app.rag. Install with: pip install faiss-cpu"
    ) from exc

from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env automatically (for CLI use)
load_dotenv()


# Default models and behavior configured here (not via .env)
DEFAULT_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
DEFAULT_CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_TOP_K = 12
DEFAULT_STRICT = False


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors / norms


def _split_text_into_chunks(
    text: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> List[str]:
    """Simple, robust text splitter by characters with overlap.

    Uses paragraphs as soft boundaries; falls back to character windows.
    """
    paragraphs = [p.strip() for p in text.replace("\r\n", "\n").split("\n\n")]
    paragraphs = [p for p in paragraphs if p]

    if not paragraphs:
        paragraphs = [text]

    chunks: List[str] = []
    buffer: List[str] = []
    current_len = 0

    for para in paragraphs:
        if current_len + len(para) + 2 <= chunk_size:
            buffer.append(para)
            current_len += len(para) + 2
        else:
            if buffer:
                chunks.append("\n\n".join(buffer))
            # start a new buffer; if paragraph is too large, hard-split it
            if len(para) <= chunk_size:
                buffer = [para]
                current_len = len(para)
            else:
                start = 0
                while start < len(para):
                    end = min(start + chunk_size, len(para))
                    chunks.append(para[start:end])
                    if end == len(para):
                        break
                    start = max(end - chunk_overlap, 0)
                buffer = []
                current_len = 0

    if buffer:
        chunks.append("\n\n".join(buffer))

    # second pass to ensure overlap
    overlapped: List[str] = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            overlapped.append(chunk)
        else:
            prefix = chunks[i - 1][-chunk_overlap:] if chunk_overlap > 0 else ""
            overlapped.append((prefix + chunk)[-chunk_size:])
    return overlapped if overlapped else chunks


class FaissRag:
    """FAISS-backed retrieval and RAG chat utilities.

    Artifacts in out_dir:
    - index.faiss: FAISS index storing normalized vectors (Inner Product)
    - docs.jsonl: one JSON line per chunk: {"id", "text"}
    - info.json: {"embedding_model", "dim"}
    - embeddings.npy: optional, saved for debug/portability
    """

    def __init__(self, out_dir: str) -> None:
        self.out_dir = Path(out_dir)
        self.index_path = self.out_dir / "index.faiss"
        self.docs_path = self.out_dir / "docs.jsonl"
        self.info_path = self.out_dir / "info.json"
        self.embeddings_path = self.out_dir / "embeddings.npy"

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.embedding_model = DEFAULT_EMBEDDING_MODEL
        self.index: Optional[faiss.Index] = None
        self.doc_texts: List[str] = []

    # -------------------- Indexing --------------------
    def build_from_text(self, text: str, chunk_size: int = 1200, chunk_overlap: int = 200) -> None:
        chunks = _split_text_into_chunks(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not chunks:
            raise ValueError("No text to index after chunking")

        vectors = self._embed_texts(chunks)
        vectors = _normalize_vectors(vectors)

        dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vectors)

        self.out_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))

        with self.docs_path.open("w", encoding="utf-8") as f:
            for i, ch in enumerate(chunks):
                f.write(json.dumps({"id": i, "text": ch}, ensure_ascii=False) + "\n")

        info = {"embedding_model": self.embedding_model, "dim": int(dim)}
        self.info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

        np.save(self.embeddings_path, vectors)
        self.doc_texts = chunks

    def build_from_file(self, input_path: str, chunk_size: int = 1200, chunk_overlap: int = 200) -> None:
        text = Path(input_path).read_text(encoding="utf-8")
        self.build_from_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        # OpenAI embeddings API accepts up to ~8192 inputs depending on model; batch to be safe
        batch_size = 512
        all_vectors: List[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            resp = self.client.embeddings.create(model=self.embedding_model, input=batch)
            vectors = np.array([d.embedding for d in resp.data], dtype=np.float32)
            all_vectors.append(vectors)
        return np.vstack(all_vectors)

    # -------------------- Load/Query --------------------
    def load(self) -> None:
        if not self.index_path.exists() or not self.docs_path.exists() or not self.info_path.exists():
            raise FileNotFoundError("Index artifacts not found. Run `index` first.")
        self.index = faiss.read_index(str(self.index_path))
        self.doc_texts = [json.loads(line)["text"] for line in self.docs_path.read_text(encoding="utf-8").splitlines()]

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[Tuple[int, float, str]]:
        if self.index is None:
            self.load()
        # Widen recall, then re-rank with keyword overlap
        query_vec = self._embed_texts([query])
        query_vec = _normalize_vectors(query_vec)
        ntotal = len(self.doc_texts)
        initial_k = min(max(top_k * 10, 50), ntotal)
        scores, idxs = self.index.search(query_vec, initial_k)

        # Keyword overlap score
        q_tokens = [t for t in ''.join([c.lower() if c.isalnum() or c.isspace() else ' ' for c in query]).split() if t]
        q_set = set(q_tokens)
        results_raw: List[Tuple[int, float, str, float]] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            if idx < 0 or idx >= len(self.doc_texts):
                continue
            text = self.doc_texts[idx]
            t_tokens = [t for t in ''.join([c.lower() if c.isalnum() or c.isspace() else ' ' for c in text]).split() if t]
            t_set = set(t_tokens)
            overlap = len(q_set.intersection(t_set))
            overlap_norm = overlap / (len(q_set) + 1e-6)
            # Phrase boost if exact phrase appears
            phrase_boost = 0.1 if query.strip().lower() in text.lower() else 0.0
            combined = 0.85 * float(score) + 0.15 * overlap_norm + phrase_boost
            results_raw.append((int(idx), combined, text, float(score)))

        # Re-rank by combined score
        results_raw.sort(key=lambda x: x[1], reverse=True)
        top = results_raw[:top_k]
        results: List[Tuple[int, float, str]] = [(idx, comb, text) for idx, comb, text, _orig in top]
        return results

    def build_context(self, query: str, top_k: int = DEFAULT_TOP_K) -> str:
        hits = self.retrieve(query, top_k=top_k)
        parts = []
        for rank, (idx, score, text) in enumerate(hits, start=1):
            parts.append(f"[Chunk {idx} | score={score:.3f}]\n{text}")
        return "\n\n".join(parts)

    # -------------------- Chat --------------------
    def chat_with_rag(
        self,
        user_message: str,
        model: str = DEFAULT_CHAT_MODEL,
        temperature: float = 0.3,
        top_k: int = DEFAULT_TOP_K,
        strict: bool = DEFAULT_STRICT,
    ) -> str:
        """Chat completion constrained by retrieved context.

        strict=True forces the model to answer only from the provided context.
        """
        context = self.build_context(user_message, top_k=top_k)
        if strict and not context.strip():
            return "I don't know based on the available knowledge."

        if strict:
            system_prompt = (
                "You must answer ONLY using the provided CONTEXT. "
                "If the answer is not contained in the context, say you don't know. Be concise.\n\n"
                f"CONTEXT:\n{context}"
            )
        else:
            system_prompt = (
                "Use the following CONTEXT to answer the user's question. Prefer facts from the CONTEXT. "
                "If some details are not explicitly covered, you may answer briefly using reasonable general knowledge and inference. "
                "Be concise and accurate.\n\n"
                f"CONTEXT:\n{context}"
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        resp = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=500,
        )
        return resp.choices[0].message.content or ""


# -------------------- CLI --------------------

def _cmd_index(args: argparse.Namespace) -> None:
    rag = FaissRag(args.out_dir)
    if args.input:
        rag.build_from_file(args.input, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    else:
        # read from stdin
        stdin_text = Path(args.stdin_text).read_text(encoding="utf-8") if args.stdin_text else None
        if stdin_text is None:
            stdin_text = input()
        rag.build_from_text(stdin_text, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    print(f"Index written to: {args.out_dir}")


def _cmd_query(args: argparse.Namespace) -> None:
    rag = FaissRag(args.out_dir)
    rag.load()
    if args.chat:
        reply = rag.chat_with_rag(
            user_message=args.q,
            model=args.model,
            temperature=args.temperature,
            top_k=args.top_k,
            strict=not args.lenient,
        )
        print(reply)
    else:
        hits = rag.retrieve(args.q, top_k=args.top_k)
        for idx, (chunk_id, score, text) in enumerate(hits, start=1):
            print(f"#{idx} chunk={chunk_id} score={score:.3f}\n{text}\n---")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("FAISS RAG utilities")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="Build FAISS index from text or file")
    p_index.add_argument("--input", type=str, default=None, help="Path to input .txt/.md/.json/.jsonl; if omitted, use --stdin-text or stdin")
    p_index.add_argument("--stdin-text", type=str, default=None, help="Optional path to a text file to be treated as stdin content")
    p_index.add_argument("--out-dir", type=str, default="./rag_index", help="Output directory for index artifacts")
    p_index.add_argument("--chunk-size", type=int, default=1200)
    p_index.add_argument("--chunk-overlap", type=int, default=200)
    p_index.set_defaults(func=_cmd_index)

    p_query = sub.add_parser("query", help="Query the FAISS index; optionally run chat with RAG")
    p_query.add_argument("--out-dir", type=str, default="./rag_index")
    p_query.add_argument("--q", type=str, required=True, help="User query")
    p_query.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p_query.add_argument("--chat", action="store_true", help="If set, run chat completion with retrieved context")
    p_query.add_argument("--model", type=str, default=DEFAULT_CHAT_MODEL)
    p_query.add_argument("--temperature", type=float, default=0.3)
    p_query.add_argument("--lenient", action="store_true", help="If set, allow answering even when context is weak")
    p_query.set_defaults(func=_cmd_query, lenient=not DEFAULT_STRICT)

    return p


def main() -> None:  # pragma: no cover
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is required in environment to run RAG.")
    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main() 