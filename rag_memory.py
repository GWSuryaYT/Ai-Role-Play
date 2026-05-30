# rag_memory.py
# Full RAG memory system using ChromaDB + Ollama nomic-embed-text embeddings.
#
# Three collections:
#   episodic   — chunks of past conversation, semantically searchable
#   lore       — world entities, facts, quest info
#   npc        — NPC personality + interaction memories
#
# How it plugs into the pipeline:
#   - Sleep cycle writes to all three collections after every N messages
#   - run_chat_turn queries relevant collections before building context
#   - Replaces the keyword-based librarian for semantic recall

import json
import os
import hashlib
import time
from typing import Optional

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_OK = True
except ImportError:
    CHROMA_OK = False
    print("[RAG] chromadb not installed. Run: pip install chromadb")

try:
    import ollama as _ollama
    OLLAMA_OK = True
except ImportError:
    OLLAMA_OK = False

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_DIR      = "data_chroma"          # persisted to disk here
EMBED_MODEL     = "nomic-embed-text"     # pull with: ollama pull nomic-embed-text
EPISODIC_COL    = "episodic_memory"      # past conversation chunks
LORE_COL        = "lore_facts"           # world entities / quest facts
NPC_COL         = "npc_memory"           # NPC personality + interactions

# How many results to retrieve per query
TOP_K_EPISODIC  = 4   # past conversation moments
TOP_K_LORE      = 3   # world/entity facts
TOP_K_NPC       = 2   # NPC memories per character

# Chunk size for episodic memory (in message pairs)
EPISODE_CHUNK   = 3   # group every 3 user+assistant pairs into one chunk


# ── Client singleton ──────────────────────────────────────────────────────────
_client: Optional["chromadb.PersistentClient"] = None

def get_client():
    global _client
    if not CHROMA_OK:
        return None
    if _client is None:
        os.makedirs(CHROMA_DIR, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False)
        )
    return _client

def get_collection(name: str):
    client = get_client()
    if client is None:
        return None
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"}   # cosine similarity for text
    )


# ── Embeddings ────────────────────────────────────────────────────────────────

def embed(text: str) -> list[float]:
    """
    Get embedding vector from Ollama nomic-embed-text.
    Falls back to a simple hash-based dummy vector if Ollama is unavailable
    (so the app doesn't crash, but RAG won't be semantic).
    """
    if not OLLAMA_OK:
        return _dummy_embed(text)
    try:
        resp = _ollama.embeddings(model=EMBED_MODEL, prompt=text)
        return resp["embedding"]
    except Exception as e:
        print(f"[RAG] Embedding failed ({e}) — using fallback")
        return _dummy_embed(text)

def embed_batch(texts: list[str]) -> list[list[float]]:
    return [embed(t) for t in texts]

def _dummy_embed(text: str) -> list[float]:
    """Deterministic 384-dim fallback when Ollama is down."""
    h = hashlib.sha256(text.encode()).digest()
    vec = [(b / 255.0) * 2 - 1 for b in h]
    return (vec * 12)[:384]   # repeat to 384 dims


# ── ID helpers ────────────────────────────────────────────────────────────────

def _make_id(prefix: str, content: str) -> str:
    """Stable unique ID from content hash."""
    h = hashlib.md5(content.encode()).hexdigest()[:12]
    return f"{prefix}_{h}"

def _ts() -> str:
    return str(int(time.time()))


# ══════════════════════════════════════════════════════════════════════════════
# WRITE OPERATIONS (called from sleep cycle)
# ══════════════════════════════════════════════════════════════════════════════

def index_episodic(history: list[dict]):
    """
    Chunk the conversation history into groups of EPISODE_CHUNK message-pairs
    and upsert each chunk into the episodic collection.
    
    Chunking strategy: pairs of (user, assistant) messages so each chunk
    is a complete micro-scene, not a half-sentence.
    """
    col = get_collection(EPISODIC_COL)
    if col is None:
        return

    # Build pairs
    pairs = []
    i = 0
    while i < len(history) - 1:
        if history[i]["role"] == "user" and history[i+1]["role"] == "assistant":
            pairs.append((history[i]["content"], history[i+1]["content"]))
            i += 2
        else:
            i += 1

    if not pairs:
        return

    # Group pairs into chunks
    chunks = []
    for j in range(0, len(pairs), EPISODE_CHUNK):
        group = pairs[j:j + EPISODE_CHUNK]
        text = ""
        for u, a in group:
            text += f"Player: {u}\nNarrator: {a}\n\n"
        text = text.strip()
        chunks.append({
            "id":       _make_id("ep", text),
            "text":     text,
            "meta":     {"chunk_index": j, "pair_count": len(group), "ts": _ts()}
        })

    if not chunks:
        return

    embeddings = embed_batch([c["text"] for c in chunks])

    col.upsert(
        ids        = [c["id"]   for c in chunks],
        documents  = [c["text"] for c in chunks],
        embeddings = embeddings,
        metadatas  = [c["meta"] for c in chunks],
    )
    print(f"[RAG] Indexed {len(chunks)} episodic chunks ({len(pairs)} pairs total)")


def index_lore(world_state: dict):
    """
    Flatten world_state entities into individual fact strings and upsert.
    Each entity becomes one document. Player status becomes one document.
    World conditions become individual documents.
    """
    col = get_collection(LORE_COL)
    if col is None:
        return

    docs = []

    # Player status as one fact block
    ps = world_state.get("player_status", {})
    if ps:
        text = (
            f"Player is currently at: {ps.get('current_location', 'Unknown')}. "
            f"Inventory: {', '.join(ps.get('inventory', [])) or 'empty'}. "
            f"Active quests: {', '.join(ps.get('active_quests', [])) or 'none'}."
        )
        docs.append({
            "id":   "lore_player_status",
            "text": text,
            "meta": {"type": "player_status", "ts": _ts()}
        })

    # Each entity as its own document
    for name, data in world_state.get("entities", {}).items():
        text = f"{name}: {json.dumps(data)}"
        docs.append({
            "id":   _make_id("lore", name),
            "text": text,
            "meta": {"type": "entity", "name": name, "ts": _ts()}
        })

    # World conditions
    for k, v in world_state.get("world_conditions", {}).items():
        text = f"World condition — {k}: {v}"
        docs.append({
            "id":   _make_id("lore_cond", k),
            "text": text,
            "meta": {"type": "world_condition", "key": k, "ts": _ts()}
        })

    if not docs:
        return

    embeddings = embed_batch([d["text"] for d in docs])
    col.upsert(
        ids        = [d["id"]   for d in docs],
        documents  = [d["text"] for d in docs],
        embeddings = embeddings,
        metadatas  = [d["meta"] for d in docs],
    )
    print(f"[RAG] Indexed {len(docs)} lore facts")


def index_npc(npc_memory: dict):
    """
    Each NPC becomes one document containing all their known facts.
    Upsert on every sleep cycle so memory stays fresh.
    """
    col = get_collection(NPC_COL)
    if col is None:
        return

    docs = []
    for name, data in npc_memory.get("npcs", {}).items():
        text = (
            f"NPC: {name}. "
            f"Personality: {data.get('personality', 'unknown')}. "
            f"Last seen at: {data.get('last_seen_location', 'unknown')}. "
            f"Relationship with player: {data.get('relationship', 'neutral')}. "
            f"Topics discussed: {data.get('topics_discussed', 'none')}. "
            f"Last interaction: {data.get('last_interaction_summary', 'no record')}."
        )
        docs.append({
            "id":   _make_id("npc", name),
            "text": text,
            "meta": {"type": "npc", "name": name, "ts": _ts()}
        })

    if not docs:
        return

    embeddings = embed_batch([d["text"] for d in docs])
    col.upsert(
        ids        = [d["id"]   for d in docs],
        documents  = [d["text"] for d in docs],
        embeddings = embeddings,
        metadatas  = [d["meta"] for d in docs],
    )
    print(f"[RAG] Indexed {len(docs)} NPC memory entries")


# ══════════════════════════════════════════════════════════════════════════════
# READ OPERATIONS (called from run_chat_turn)
# ══════════════════════════════════════════════════════════════════════════════

def query_episodic(query_text: str, top_k: int = TOP_K_EPISODIC) -> list[str]:
    """
    Find past conversation moments semantically similar to the current query.
    Returns list of text chunks, most relevant first.
    Skips results with distance > 0.55 (too dissimilar to be useful).
    """
    col = get_collection(EPISODIC_COL)
    if col is None or col.count() == 0:
        return []

    vec = embed(query_text)
    try:
        results = col.query(
            query_embeddings = [vec],
            n_results        = min(top_k, col.count()),
            include          = ["documents", "distances"]
        )
    except Exception as e:
        print(f"[RAG] Episodic query failed: {e}")
        return []

    docs      = results["documents"][0]
    distances = results["distances"][0]

    # Filter by similarity threshold (cosine distance — lower = more similar)
    return [d for d, dist in zip(docs, distances) if dist < 0.55]


def query_lore(query_text: str, top_k: int = TOP_K_LORE) -> list[str]:
    """
    Find world facts / entity data relevant to the query.
    """
    col = get_collection(LORE_COL)
    if col is None or col.count() == 0:
        return []

    vec = embed(query_text)
    try:
        results = col.query(
            query_embeddings = [vec],
            n_results        = min(top_k, col.count()),
            include          = ["documents", "distances"]
        )
    except Exception as e:
        print(f"[RAG] Lore query failed: {e}")
        return []

    docs      = results["documents"][0]
    distances = results["distances"][0]
    return [d for d, dist in zip(docs, distances) if dist < 0.50]


def query_npc(query_text: str, top_k: int = TOP_K_NPC) -> list[str]:
    """
    Find NPC memory entries relevant to the query.
    Lower threshold (0.45) because NPC retrieval should be precise.
    """
    col = get_collection(NPC_COL)
    if col is None or col.count() == 0:
        return []

    vec = embed(query_text)
    try:
        results = col.query(
            query_embeddings = [vec],
            n_results        = min(top_k, col.count()),
            include          = ["documents", "distances"]
        )
    except Exception as e:
        print(f"[RAG] NPC query failed: {e}")
        return []

    docs      = results["documents"][0]
    distances = results["distances"][0]
    return [d for d, dist in zip(docs, distances) if dist < 0.45]


def query_all(query_text: str) -> dict:
    """
    Master query — hits all three collections at once.
    Returns dict with keys: episodic, lore, npc
    """
    return {
        "episodic": query_episodic(query_text),
        "lore":     query_lore(query_text),
        "npc":      query_npc(query_text),
    }


# ══════════════════════════════════════════════════════════════════════════════
# CONTEXT BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_rag_context(query_text: str) -> str:
    """
    Queries all collections and builds a clean context string for injection
    into the system prompt.
    
    Returns empty string if nothing relevant found (no noise injection).
    """
    results = query_all(query_text)

    episodic = results["episodic"]
    lore     = results["lore"]
    npc      = results["npc"]

    if not any([episodic, lore, npc]):
        return ""

    ctx = ""

    if lore:
        ctx += "[WORLD KNOWLEDGE]\n"
        for fact in lore:
            ctx += f"- {fact}\n"
        ctx += "\n"

    if npc:
        ctx += "[CHARACTER MEMORY]\n"
        for mem in npc:
            ctx += f"- {mem}\n"
        ctx += "\n"

    if episodic:
        ctx += "[RELEVANT PAST MOMENTS]\n"
        for chunk in episodic:
            ctx += f"---\n{chunk}\n"
        ctx += "\n"

    return ctx.strip()


# ══════════════════════════════════════════════════════════════════════════════
# ADMIN / UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def get_stats() -> dict:
    """Returns count of documents in each collection."""
    stats = {}
    for name in [EPISODIC_COL, LORE_COL, NPC_COL]:
        col = get_collection(name)
        stats[name] = col.count() if col else 0
    return stats

def reset_collection(name: str):
    """Wipe a specific collection. Useful during testing."""
    client = get_client()
    if client:
        try:
            client.delete_collection(name)
            print(f"[RAG] Collection '{name}' reset.")
        except Exception:
            pass

def reset_all():
    """Wipe all RAG data. Prompts for confirmation."""
    for name in [EPISODIC_COL, LORE_COL, NPC_COL]:
        reset_collection(name)
    print("[RAG] All collections reset.")

def ensure_embed_model():
    """
    Checks if nomic-embed-text is available in Ollama.
    Prints instructions if not.
    """
    if not OLLAMA_OK:
        print("[RAG] Ollama not available.")
        return False
    try:
        _ollama.embeddings(model=EMBED_MODEL, prompt="test")
        print(f"[RAG] Embedding model '{EMBED_MODEL}' is ready.")
        return True
    except Exception:
        print(f"[RAG] Embedding model '{EMBED_MODEL}' not found.")
        print(f"[RAG] Run: ollama pull {EMBED_MODEL}")
        return False