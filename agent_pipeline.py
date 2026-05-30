# agent_pipeline.py
import ollama
import json
import os
from config import MAIN_MODEL, FAST_MODEL
from world_state import load_world_state, update_world_state, get_entity_keys
from rag_memory import (
    build_rag_context,
    index_episodic, index_lore, index_npc,
    get_stats as rag_stats,
)

PROMPTS_FILE    = "config_prompts.json"
NPC_MEMORY_FILE = "data_npc_memory.json"

DEFAULT_PROMPTS = {
    "system_prompt_template": (
        "You are the Narrator. Use this world context to inform your response "
        "but never quote it directly.\n\n{targeted_context}"
    ),
    "router_prompt": (
        "You are a smart routing agent. Decide if the user's message needs world/entity data to answer well.\n"
        "Only return YES if the message references a specific person, place, item, quest, past event, or NPC by name.\n"
        "Return NO if it's casual dialogue, simple actions, jokes, or general conversation that needs no external memory.\n\n"
        "User message: {user_input}\n\n"
        "Output strictly: {{\"needs_world_data\": true}} or {{\"needs_world_data\": false}}"
    ),
    "tone_router_prompt": (
        "You are a tone classifier for an RPG game.\n"
        "Read the user message and classify it into exactly one of these tones:\n\n"
        "  ACTION   - Player is performing a physical action (attacking, moving, casting, exploring)\n"
        "  DIALOGUE - Player is speaking to an NPC or character\n"
        "  REFLECT  - Player is thinking, planning, reading, or processing information\n"
        "  OOC      - Out of character: casual chat, hype, meta questions\n\n"
        "User message: {user_input}\n\n"
        "Output strictly: {{\"tone\": \"ACTION\"}} or {{\"tone\": \"DIALOGUE\"}} or "
        "{{\"tone\": \"REFLECT\"}} or {{\"tone\": \"OOC\"}}"
    ),
    "librarian_prompt": (
        "You are a librarian. Identify which Known Entities are relevant to the User Input.\n"
        "Known Entities: {entities_list}\n"
        "User Input: {user_input}\n"
        "Output strictly pure JSON: {{\"relevant\": [\"Name1\", \"Name2\"]}}"
    ),
    "sleep_cycle_prompt": (
        "You are the World Simulation Engine. Review the recent chat log and update the World JSON State.\n"
        "CRITICAL FILTER: Be highly selective. Ignore minor actions, flavor text, and background scenery.\n"
        "ONLY save or update essential narrative elements like: Persons, Places, Times/Dates, Active Quests, Imminent Dangers.\n\n"
        "Rules:\n"
        "1. Maintain total schema freedom to add nested objects under 'player_status' or 'entities'.\n"
        "2. Preserve existing data unless directly changed by recent events.\n"
        "3. For NPCs/characters you interact with, update their entry with: personality traits observed, "
        "topics discussed, current relationship status, and last interaction summary.\n"
        "4. Track WORLD CONDITIONS: wars, disasters, political states, active threats.\n\n"
        "[CURRENT STATE]:\n{current_state}\n\n"
        "[RECENT CHAT]:\n{chat_log}\n\n"
        "Output the ENTIRE updated JSON state. No extra text allowed."
    ),
    "npc_memory_prompt": (
        "You are analyzing a conversation to extract NPC/character memory data.\n"
        "Review the chat and identify any NPCs or named characters that appeared.\n"
        "For each NPC found, extract:\n"
        "- personality: key traits observed (max 3 bullet points)\n"
        "- last_seen_location: where they were\n"
        "- topics_discussed: what was talked about (brief)\n"
        "- relationship: how they feel about the player (neutral/friendly/hostile/etc)\n"
        "- last_interaction_summary: 1-2 sentence summary of what happened\n\n"
        "[CURRENT NPC MEMORY]:\n{current_npc_memory}\n\n"
        "[RECENT CHAT]:\n{chat_log}\n\n"
        "Output strictly JSON: {{\"npcs\": {{\"NPC Name\": {{...fields...}}}}}}\n"
        "Only include NPCs that actually appeared in this chat."
    )
}

TONE_INSTRUCTIONS = {
    "ACTION": (
        "The player is performing an action. Respond with vivid atmospheric narration. "
        "Describe outcomes, sensations, consequences. Stay in second-person. "
        "No meta commentary. No lists."
    ),
    "DIALOGUE": (
        "The player is speaking to a character. Respond AS that character or narrate the exchange. "
        "Stay fully in-world. Voice the NPC naturally based on their personality. "
        "Remember prior interactions from the memory context."
    ),
    "REFLECT": (
        "The player is thinking, reading, or planning inside the world. "
        "Respond with immersive inner-world narration. "
        "Keep it grounded and atmospheric, not melodramatic."
    ),
    "OOC": (
        "The player is speaking out of character - just chatting or reacting casually. "
        "Respond conversationally and warmly as their game companion. "
        "Do NOT write dramatic narration. Keep it short, natural, friendly. "
        "Match their energy."
    ),
}


# --- File I/O -----------------------------------------------------------------

def load_prompts():
    if not os.path.exists(PROMPTS_FILE):
        with open(PROMPTS_FILE, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_PROMPTS, f, indent=4)
        return DEFAULT_PROMPTS
    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    for k, v in DEFAULT_PROMPTS.items():
        if k not in data:
            data[k] = v
    return data

def save_prompts(prompts_dict):
    with open(PROMPTS_FILE, "w", encoding="utf-8") as f:
        json.dump(prompts_dict, f, indent=4)

def load_npc_memory():
    if not os.path.exists(NPC_MEMORY_FILE):
        return {"npcs": {}}
    with open(NPC_MEMORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_npc_memory(data):
    with open(NPC_MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# --- Core LLM -----------------------------------------------------------------

def generate_chat(messages, model=MAIN_MODEL):
    response = ollama.chat(model=model, messages=messages)
    return response['message']['content']

def _fast_json_call(prompt):
    try:
        response = ollama.generate(model=FAST_MODEL, prompt=prompt, format="json")
        return json.loads(response['response'])
    except Exception:
        return None


# --- Tone Router --------------------------------------------------------------

def tone_router(user_input):
    prompts = load_prompts()
    prompt  = prompts["tone_router_prompt"].format(user_input=user_input)
    result  = _fast_json_call(prompt)
    if result is None:
        return "REFLECT"
    tone = result.get("tone", "REFLECT").upper()
    return tone if tone in TONE_INSTRUCTIONS else "REFLECT"


# --- World Data Router --------------------------------------------------------

def router_agent(user_input):
    prompts = load_prompts()
    prompt  = prompts["router_prompt"].format(user_input=user_input)
    result  = _fast_json_call(prompt)
    if result is None:
        return False
    return result.get("needs_world_data", False)


# --- Librarian (structured fallback) -----------------------------------------

def librarian_agent(user_input, recent_chat):
    known_entities = get_entity_keys()
    npc_memory     = load_npc_memory()
    known_npcs     = list(npc_memory.get("npcs", {}).keys())
    all_known      = list(set(known_entities + known_npcs))
    if not all_known:
        return [], []
    prompts = load_prompts()
    prompt  = prompts["librarian_prompt"].format(
        entities_list=", ".join(all_known),
        user_input=user_input
    )
    result = _fast_json_call(prompt)
    if result is None:
        return [], []
    relevant       = result.get("relevant", [])
    relevant_world = [r for r in relevant if r in known_entities]
    relevant_npcs  = [r for r in relevant if r in known_npcs]
    return relevant_world, relevant_npcs


# --- Structured Context Builder -----------------------------------------------

def build_structured_context(relevant_world_entities, relevant_npc_names):
    from world_state import get_targeted_context as _world_ctx
    context = ""
    if relevant_world_entities:
        context += _world_ctx(relevant_world_entities)
    if relevant_npc_names:
        npc_data = load_npc_memory().get("npcs", {})
        context += "\n[NPC MEMORY FILES]\n"
        for name in relevant_npc_names:
            npc = npc_data.get(name)
            if npc:
                context += f"--- {name} ---\n"
                context += f"  Personality: {npc.get('personality', 'Unknown')}\n"
                context += f"  Last seen at: {npc.get('last_seen_location', 'Unknown')}\n"
                context += f"  Relationship: {npc.get('relationship', 'Neutral')}\n"
                context += f"  Topics discussed: {npc.get('topics_discussed', 'None')}\n"
                context += f"  Last interaction: {npc.get('last_interaction_summary', 'No prior meeting')}\n\n"
    if not context:
        context = _world_ctx([])
    return context


# --- History Cleaner ----------------------------------------------------------

def get_clean_history(full_history, current_user_text, window=8):
    trimmed = full_history[:]
    if trimmed and trimmed[-1].get("role") == "user" and trimmed[-1].get("content") == current_user_text:
        trimmed = trimmed[:-1]
    return trimmed[-window:]


# --- Sleep Cycle --------------------------------------------------------------

def run_sleep_cycle(episodic_memory):
    chat_log      = "\n".join([f"{m['role']}: {m['content']}" for m in episodic_memory[-6:]])
    current_state = load_world_state()
    prompts       = load_prompts()

    # Task 1: World State
    world_prompt = prompts["sleep_cycle_prompt"].format(
        current_state=json.dumps(current_state, indent=2),
        chat_log=chat_log
    )
    new_state = current_state
    try:
        print("\n[BACKGROUND] Sleep Cycle: World State...")
        response  = ollama.generate(model=MAIN_MODEL, prompt=world_prompt, format="json")
        parsed    = json.loads(response['response'])
        if "player_status" in parsed and "entities" in parsed:
            update_world_state(parsed)
            new_state = parsed
            print("[BACKGROUND] World state updated.")
    except Exception as e:
        print(f"[BACKGROUND] World State failed: {e}")

    # Task 2: NPC Memory
    current_npc_memory = load_npc_memory()
    npc_prompt = prompts["npc_memory_prompt"].format(
        current_npc_memory=json.dumps(current_npc_memory, indent=2),
        chat_log=chat_log
    )
    updated_npc_memory = current_npc_memory
    try:
        print("[BACKGROUND] Sleep Cycle: NPC Memory...")
        response   = ollama.generate(model=FAST_MODEL, prompt=npc_prompt, format="json")
        npc_update = json.loads(response['response'])
        if "npcs" in npc_update:
            merged = current_npc_memory.get("npcs", {})
            merged.update(npc_update["npcs"])
            updated_npc_memory = {"npcs": merged}
            save_npc_memory(updated_npc_memory)
            print(f"[BACKGROUND] NPC memory updated: {list(npc_update['npcs'].keys())}")
    except Exception as e:
        print(f"[BACKGROUND] NPC Memory failed: {e}")

    # Task 3: RAG Index Update
    print("[BACKGROUND] Sleep Cycle: RAG indexing...")
    try:
        index_episodic(episodic_memory)
        index_lore(new_state)
        index_npc(updated_npc_memory)
        stats = rag_stats()
        print(f"[BACKGROUND] RAG indexed: {stats}")
    except Exception as e:
        print(f"[BACKGROUND] RAG indexing failed: {e}")

    return True


# --- Main Chat Pipeline -------------------------------------------------------

def run_chat_turn(user_text, history):
    """
    Pipeline:
    1. Tone router        - ACTION / DIALOGUE / REFLECT / OOC
    2. RAG semantic query - hits episodic + lore + npc collections
    3. World router       - structured JSON lookup needed?
    4. Librarian          - which entities/NPCs from JSON?
    5. Merge contexts     - RAG (semantic) + structured JSON (authoritative)
    6. Build system prompt with tone instruction + merged context
    7. Clean history      - no duplicate of current message
    8. Generate
    """
    debug = {
        "router":   "SKIP",
        "entities": [],
        "npcs":     [],
        "tone":     "REFLECT",
        "rag":      {"episodic": 0, "lore": 0, "npc": 0}
    }

    # 1. Tone
    tone          = tone_router(user_text)
    debug["tone"] = tone

    # 2. RAG query (skip for OOC)
    rag_context = ""
    if tone != "OOC":
        from rag_memory import query_all
        rag_results  = query_all(user_text)
        debug["rag"] = {
            "episodic": len(rag_results["episodic"]),
            "lore":     len(rag_results["lore"]),
            "npc":      len(rag_results["npc"]),
        }
        rag_context = build_rag_context(user_text)

    # 3 & 4. Structured JSON lookup (skip for OOC)
    structured_context = ""
    if tone != "OOC":
        needs_data      = router_agent(user_text)
        debug["router"] = "FETCH" if needs_data else "SKIP"
        if needs_data:
            relevant_world, relevant_npcs = librarian_agent(user_text, history)
            debug["entities"] = relevant_world
            debug["npcs"]     = relevant_npcs
            structured_context = build_structured_context(relevant_world, relevant_npcs)
        else:
            structured_context = build_structured_context([], [])
    else:
        debug["router"] = "SKIP (OOC)"

    # 5. Merge: RAG first (broad semantic), structured second (authoritative)
    combined_context = ""
    if rag_context:
        combined_context += rag_context + "\n\n"
    if structured_context.strip():
        combined_context += "[AUTHORITATIVE WORLD STATE - treat as ground truth]\n"
        combined_context += structured_context

    # 6. Build system prompt
    tone_instruction = TONE_INSTRUCTIONS[tone]
    if combined_context.strip():
        sys_content = (
            f"{tone_instruction}\n\n"
            "Memory and world context (weave naturally, never quote directly):\n"
            f"{combined_context.strip()}"
        )
    else:
        sys_content = tone_instruction

    # 7. Clean history
    clean_history = get_clean_history(history, user_text, window=8)

    messages = [{"role": "system", "content": sys_content}]
    messages.extend(clean_history)
    messages.append({"role": "user", "content": user_text})

    # 8. Generate
    response = generate_chat(messages)
    return response, debug