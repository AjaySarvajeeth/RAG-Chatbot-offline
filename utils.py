import re

def split_into_sentences(text: str) -> list[str]:
    """
    Offline regex-based sentence splitter.
    Works for English-like languages.
    """
    if not text:
        return []
    # Normalize newlines/spaces
    text = re.sub(r"\s*\n\s*", " ", text.strip())
    # Split on punctuation + space (basic heuristic)
    sents = re.split(r'(?<=[.!?])\s+', text)
    # Clean & filter
    return [s.strip() for s in sents if s and s.strip()]

def smart_chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 120,
    source: str = "unknown",
    lang: str = "en"
) -> list[dict]:
    """
    Chunk text into segments of ~`chunk_size` chars with `overlap`.
    Overlap is sentence-aware (carry over last few sentences rather than raw chars).
    """
    sents = split_into_sentences(text)
    chunks = []
    cur_sents = []
    cur_len = 0
    chunk_idx = 0

    # approx sentences to keep for overlap, based on chars
    keep_n = max(1, overlap // 100) if overlap > 0 else 0

    for s in sents:
        s_len = len(s) + 1  # account for space
        if cur_len + s_len <= chunk_size:
            cur_sents.append(s)
            cur_len += s_len
        else:
            if cur_sents:
                chunk_text = " ".join(cur_sents).strip()
                if chunk_text:
                    chunks.append({
                        "id": f"{source}-{chunk_idx}",
                        "text": chunk_text,
                        "position": chunk_idx,
                        "source": source,
                        "language": lang
                    })
                    chunk_idx += 1
            # sentence-level overlap carryover
            overlap_sents = cur_sents[-keep_n:] if keep_n > 0 else []
            cur_sents = overlap_sents + [s]
            cur_len = sum(len(x) for x in cur_sents) + max(len(cur_sents) - 1, 0)

    # Add remainder
    if cur_sents:
        chunk_text = " ".join(cur_sents).strip()
        if chunk_text:
            chunks.append({
                "id": f"{source}-{chunk_idx}",
                "text": chunk_text,
                "position": chunk_idx,
                "source": source,
                "language": lang
            })

    return chunks