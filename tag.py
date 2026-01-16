from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
from typing import List, Dict, Any
from openai import OpenAI
import json
from pathlib import Path

SEP_RE = re.compile(r"[\s\-_\/]+")

def normalize(s: str) -> str:
    s = s.strip()
    s = SEP_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s)
    return s

class TrieNode:
    __slots__ = ("children", "end")
    def __init__(self):
        self.children: Dict[str, "TrieNode"] = {}
        self.end = False

class CharTrie:
    """
    Character-level trie: works for both 'new balance' and 'ランニングシューズ' (no-space).
    """
    def __init__(self, phrases: Set[str]):
        self.root = TrieNode()
        for p in phrases:
            self.add(p)

    def add(self, phrase: str) -> None:
        phrase = phrase.strip()
        if not phrase:
            return
        node = self.root
        for ch in phrase:
            node = node.children.setdefault(ch, TrieNode())
        node.end = True

    def matches(self, text: str) -> List[Tuple[int, int]]:
        out: List[Tuple[int, int]] = []
        n = len(text)
        text = text.lower()
        for i in range(n):
            node = self.root
            
            j = i
            while j < n and text[j] in node.children:
                node = node.children[text[j]]
                j += 1
                if node.end:
                    out.append((i, j))
        return out


def select_non_overlapping(matches: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Greedy by (length desc, start asc), skip overlaps.
    Returns spans sorted by start (reading order).
    """
    if not matches:
        return []

    # Prefer longer spans; tie-break by earlier start
    matches_sorted = sorted(matches, key=lambda se: (-(se[1] - se[0]), se[0], se[1]))

    chosen: List[Tuple[int, int]] = []

    for s, e in matches_sorted:
        overlap = False
        for cs, ce in chosen:
            if not (e <= cs or s >= ce):
                overlap = True
                break
        if not overlap:
            chosen.append((s, e))

    chosen.sort(key=lambda se: se[0])
    return chosen



def _lower_if_latin(s: str) -> str:
    s = s.strip()
    if any("a" <= c.lower() <= "z" for c in s):
        return s.lower()
    return s


def tokenize_unified(keyword: str, trie: CharTrie) -> Tuple[List[str], List[str]]:
    """
    Example:
      text="new balance running shoes man size10"
      dict has {"running shoes"}
      -> ["new balance", "running shoes", "man size10"]
    """
    text = normalize(keyword)

    all_matches = trie.matches(text)                 # List[(start,end)]
    chosen = select_non_overlapping(all_matches)     # non-overlapping
    chosen.sort(key=lambda se: se[0])                # reading order

    tokens: List[str] = []
    cur = 0

    for s, e in chosen:
        # residual chunk before phrase
        if cur < s:
            chunk = text[cur:s].strip()
            if chunk:
                tokens.append(chunk)

        phrase = text[s:e].strip()
        if phrase:
            tokens.append(phrase)

        cur = e

    # residual chunk after last phrase
    if cur < len(text):
        tail = text[cur:].strip()
        if tail:
            tokens.append(tail)

    # normalize casing for latin
    tokens = [_lower_if_latin(t) for t in tokens if t and t.strip()]

    return tokens


@dataclass
class TagDicts:
    brands: Set[str]      # 品牌词
    products: Set[str]    # 商品词
    crowds: Set[str]       # 人群词
    scenes: Set[str]       # 场景词
    colors: Set[str]      # 颜色词
    sizes: Set[str]        # 尺寸词
    features: Set[str]    # 卖点词
    attributes: Set[str]   # 属性词 

TAG_TYPES = ["品牌词", "商品词", "人群词", "场景词", "颜色词", "尺寸词", "卖点词", "属性词"]


def tag_token(token: str, d: TagDicts) -> Tuple[List[str], float]:
    """
    Dictionary + regex tagging.
    Output tags are GUARANTEED to be subset of TAG_TYPES.
    """
    t = token.strip()
    if not t:
        return [], 0.0

    tl = t.lower()
    tags: List[str] = []

    if tl in d.brands:
        tags.append("品牌词")
    if tl in d.products:
        tags.append("商品词")
    if tl in d.crowds:
        tags.append("人群词")
    if tl in d.scenes:
        tags.append("场景词")
    if tl in d.colors:
        tags.append("颜色词")
    if tl in d.features:
        tags.append("卖点词")
    if tl in d.attributes:
        tags.append("属性词")
    if tl in d.sizes:
        tags.append("尺寸词")

    if not tags:
        return [], 0.55
    conf = 0.90

    tags = list(dict.fromkeys(tags))
    return tags, round(conf, 2)


def default_tag_dicts() -> TagDicts:
    return TagDicts(
        brands=set([]),
        products=set([]),
        crowds=set([]),
        scenes=set([]),
        colors=set([]),
        sizes=set([]),
        features=set([]),
        attributes=set([]),
    )

LLM_MIN_OVERRIDE_CONF = 0.70   
LLM_TRIGGER_CONF = 0.60       


def _should_send_to_llm(tok: str, tags: List[str], conf: float) -> bool:
    t = tok.strip()
    if not t:
        return False
    if conf < LLM_TRIGGER_CONF:
        return True
    return False


client = OpenAI()

def llm_classify_phrases_batch(phrases: List[str]) -> List[Dict]:
    """
    Returns list of:
      {"phrase": str, "canonical": str, "tags": [..], "confidence": float}
    Uses Structured Outputs (JSON Schema) so output is always valid.
    """
    phrases_norm = [p.strip().lower() for p in phrases if p and p.strip()]
    if not phrases_norm:
        return []
    
    schema = {
        "type": "object",
        "properties": {
            "tokens": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "canonical": {"type": "string"},
                        "tags": {
                            "type": "array",
                            "items": {"type": "string", "enum": TAG_TYPES},
                            "minItems": 1
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0
                        }
                    },
                    "required": ["text", "canonical", "tags", "confidence"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["tokens"],
        "additionalProperties": False
    }


    prompt = (
        "You are an ecommerce keyword tokenizer and tagger.\n\n"
        "Given an input phrase, you must:\n"
        "1) Split it into meaningful keyword tokens (phrases, not individual words).\n"
        "2) Assign one or more tags to each token from the following list:\n"
        f"{TAG_TYPES}\n\n"
        "Rules:\n"
        "- Tokens MUST preserve original order and NOT overlap.\n"
        "- Prefer multi-word phrases when they form a semantic unit (e.g. brand, product).\n"
        "- Return lowercased text.\n"
        "- Provide a normalized `canonical` form.\n"
        "- Confidence is 0.0 to 1.0.\n\n"
        "Input:\n"
        f"{phrases_norm}"
    )

    resp = client.responses.create(
        model="gpt-5",   
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": "phrase_tags",
                "schema": schema,
                "strict": True
            }
        }
    )
    data = json.loads(resp.output_text)
    return data["tokens"]



def learn_to_dictionary(d, token: str, tags: List[str]) -> None:
    """
    Add token into relevant dict sets aligned with TAG_TYPES
    """
    t = token.strip()
    if not t:
        return
    tl = t.lower()

    if "品牌词" in tags:
        d.brands.add(tl)
    if "商品词" in tags:
        d.products.add(tl)
    if "人群词" in tags:
        d.crowds.add(tl)
    if "场景词" in tags:
        d.scenes.add(tl)
    if "颜色词" in tags:
        d.colors.add(tl)
    if "尺寸词" in tags:
        d.sizes.add(tl)
    if "卖点词" in tags:
        d.features.add(tl)
    if "属性词" in tags:
        d.attributes.add(tl)

# TEMP, might change later for batch sbmit to llm
def tokenize_and_tag_batch(keywords: List[str]) -> List[Dict]:
    result = []
    d = default_tag_dicts()
    d = load_dict(d)
    phrase_set: Set[str] = (
        d.brands | d.products | d.crowds | d.scenes | d.colors | d.sizes | d.features | d.attributes
    )
    norm_phrases = {normalize(p).strip() for p in phrase_set if p and normalize(p).strip()}
    trie = CharTrie(norm_phrases)
    for key in keywords:
        result.append(tokenize_and_tag_helper(key, trie, d))
    save_dict(d)
    return result

def tokenize_and_tag(keyword: str) -> Dict:
    d = default_tag_dicts()
    d = load_dict(d)

    phrase_set: Set[str] = (
        d.brands | d.products | d.crowds | d.scenes | d.colors | d.sizes | d.features | d.attributes
    )
    norm_phrases = {normalize(p).strip() for p in phrase_set if p and normalize(p).strip()}
    trie = CharTrie(norm_phrases)

    result = tokenize_and_tag_helper(keyword, trie, d)
    save_dict(d)
    return result

def tokenize_and_tag_helper(keyword: str, trie: CharTrie, d: TagDicts) -> Dict:
    tokens = tokenize_unified(keyword, trie)

    tagged_tokens: List[Dict] = []
    tag_summary: Dict[str, List[str]] = {k: [] for k in TAG_TYPES}

    token_meta = []
    unknown_tokens = []
    for tok in tokens:
        tags, conf = tag_token(tok, d)
        if _should_send_to_llm(tok, tags, conf):
            unknown_tokens.append(tok)
        else:
            token_meta.append((tok, tags, conf))
    llm_tokens = []
    if unknown_tokens:
        llm_results = llm_classify_phrases_batch(unknown_tokens)
        for r in llm_results:
            text = (r.get("text") or "").strip().lower()
            if not text:
                continue
            token_meta.append((text, r.get("tags") or [], float(r.get("confidence") or 0.0)))
            llm_tokens.append(text)
    llm_tokens = set(llm_tokens)
    tokens = []
    for tok, tags, conf in token_meta:
        tokens.append(tok)
        key = tok.strip().lower()
        if key in llm_tokens and conf >= 0.85:
            learn_to_dictionary(d, tok, tags)
            trie.add(tok)
        tagged_tokens.append({"token": tok, "tags": tags, "confidence": round(conf, 2)})
        for t in tags:
            if tok not in tag_summary[t]:
                tag_summary[t].append(tok)

    tag_summary = {k: v for k, v in tag_summary.items() if v}
    return {
        "original_keyword": keyword,
        "tokens": tokens,
        "tagged_tokens": tagged_tokens,
        "tag_summary": tag_summary,
    }

DICT_PATH = Path("keyword_dict.json")

def save_dict(d):
    with open(DICT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "brands": sorted(d.brands),
                "products": sorted(d.products),
                "crowds": sorted(d.crowds),
                "scenes": sorted(d.scenes),
                "colors": sorted(d.colors),
                "sizes": sorted(d.sizes),
                "features": sorted(d.features),
                "attributes": sorted(d.attributes),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

def load_dict(d):
    if not DICT_PATH.exists():
        return d

    with open(DICT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    d.brands |= set(data.get("brands", []))
    d.products |= set(data.get("products", []))
    d.crowds |= set(data.get("crowds", []))
    d.scenes |= set(data.get("scenes", []))
    d.colors |= set(data.get("colors", []))
    d.sizes |= set(data.get("sizes", []))
    d.features |= set(data.get("features", []))
    d.attributes |= set(data.get("attributes", []))

    return d


if __name__ == "__main__":
    tests = [
        "ランニング リュック 軽量 小さめ",
    ]
    for s in tests:
        print(json.dumps(tokenize_and_tag(s), ensure_ascii=False, indent=2))
