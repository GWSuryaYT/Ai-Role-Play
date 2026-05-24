# agent_pipeline.py
import ollama
import json
import os
from config import MAIN_MODEL, FAST_MODEL
from world_state import load_world_state, update_world_state, get_entity_keys

PROMPTS_FILE = "config_prompts.json"
NPC_MEMORY_FILE = "data_npc_memory.json"

DEFAULT_PROMPTS = {
    "system_prompt_template": (
        "You are an immersive RPG narrator and character simulator. Stay in character at all times.\n"
        "Maintain narrative continuity and never contradict established facts.\n\n"
        "{targeted_context}"
    ),
    "router_prompt": (
        "You are a smart routing agent. Decide if the user's message needs world/entity data to answer well.\n"
        "Only return YES if the message references a specific person, place, item, quest, past event, or NPC by name.\n"
        "Return NO if it's casual dialogue, simple actions, jokes, or general conversation that needs no external memory.\n\n"
        "User message: {user_input}\n\n"
        "Output strictly: {{\"needs_world_data\": true}} or {{\"needs_world_data\": false}}"
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
        "ONLY save or update essential narrative elements like: Persons, Places, Times/Dates, Active Quests, and Imminent Dangers.\n\n"
        "Rules:\n"
        "1. Maintain total schema freedom to add nested objects under 'player_status' or 'entities'.\n"
        "2. Preserve existing data unless directly changed by recent events.\n"
        "3. For NPCs/characters you interact with, update their entry with: personality traits observed, "
        "topics discussed, current relationship status, and last interaction summary.\n"
        "4. Track WORLD CONDITIONS: wars, disasters, political states, active threats that would affect NPC behavior.\n\n"
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
        "Only include NPCs that actually appeared in this chat. Preserve existing NPC data for NPCs not in this chat."
    )
}


# ─── File I/O ──────────────────────────────────────────────────────────────────

def load_prompts():
    if not os.path.exists(PROMPTS_FILE):
        with open(PROMPTS_FILE, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_PROMPTS, f, indent=4)
        return DEFAULT_PROMPTS
    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Backfill any missing keys from defaults
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


# ─── Core LLM Calls ────────────────────────────────────────────────────────────

def generate_chat(messages, model=MAIN_MODEL):
    response = ollama.chat(model=model, messages=messages)
    return response['message']['content']

def _fast_json_call(prompt):
    """Helper: call fast model, return parsed JSON or None."""
    try:
        response = ollama.generate(model=FAST_MODEL, prompt=prompt, format="json")
        return json.loads(response['response'])
    except Exception:
        return None


# ─── Router Agent ──────────────────────────────────────────────────────────────

def router_agent(user_input):
    """
    Stage 1: Decide if world data is needed AT ALL.
    Returns True if we should fetch from world state, False if we can skip entirely.
    This prevents burning context on casual messages like 'lol ok' or 'tell me a joke'.
    """
    prompts = load_prompts()
    prompt = prompts["router_prompt"].format(user_input=user_input)
    result = _fast_json_call(prompt)
    if result is None:
        return False  # If routing fails, skip world data rather than waste tokens
    return result.get("needs_world_data", False)


# ─── Librarian Agent ───────────────────────────────────────────────────────────

def librarian_agent(user_input, recent_chat):
    """
    Stage 2 (only runs if router said YES): 
    Find which specific entities from world state are relevant.
    Also checks NPC memory for known characters.
    """
    known_entities = get_entity_keys()
    npc_memory = load_npc_memory()
    known_npcs = list(npc_memory.get("npcs", {}).keys())
    
    all_known = list(set(known_entities + known_npcs))
    if not all_known:
        return [], []

    prompts = load_prompts()
    prompt = prompts["librarian_prompt"].format(
        entities_list=", ".join(all_known),
        user_input=user_input
    )
    result = _fast_json_call(prompt)
    if result is None:
        return [], []
    
    relevant = result.get("relevant", [])
    
    # Split into world entities vs NPC names
    relevant_world = [r for r in relevant if r in known_entities]
    relevant_npcs  = [r for r in relevant if r in known_npcs]
    return relevant_world, relevant_npcs


# ─── Context Builder ───────────────────────────────────────────────────────────

def build_targeted_context(relevant_world_entities, relevant_npc_names):
    """
    Assembles a lean context string from only what was requested.
    Pulls from world_state for locations/quests/facts,
    and from npc_memory for character personalities and past interactions.
    """
    from world_state import get_targeted_context as _world_ctx
    context = ""

    # World data (locations, quests, items, etc.)
    if relevant_world_entities:
        context += _world_ctx(relevant_world_entities)

    # NPC memory (who they are, what was said last time)
    if relevant_npc_names:
        npc_data = load_npc_memory().get("npcs", {})
        context += "\n[NPC MEMORY FILES]\n"
        for name in relevant_npc_names:
            npc = npc_data.get(name)
            if npc:
                context += f"--- {name} ---\n"
                context += f"  Personality: {npc.get('personality', 'Unknown')}\n"
                context += f"  Last seen at: {npc.get('last_seen_location', 'Unknown')}\n"
                context += f"  Relationship to player: {npc.get('relationship', 'Neutral')}\n"
                context += f"  Topics discussed before: {npc.get('topics_discussed', 'None')}\n"
                context += f"  Last interaction: {npc.get('last_interaction_summary', 'No prior meeting')}\n\n"

    # Always include player status (tiny, always relevant)
    if not context:
        context = _world_ctx([])  # Just player status, no entities

    return context


# ─── Sleep Cycle ───────────────────────────────────────────────────────────────

def run_sleep_cycle(episodic_memory):
    """
    Runs two background tasks:
    1. Update world state (locations, quests, world conditions)
    2. Update NPC memory (personalities, what was discussed, relationships)
    """
    chat_log = "\n".join([f"{m['role']}: {m['content']}" for m in episodic_memory[-6:]])
    current_state = load_world_state()
    prompts = load_prompts()

    # ── Task 1: World State Update ──
    world_prompt = prompts["sleep_cycle_prompt"].format(
        current_state=json.dumps(current_state, indent=2),
        chat_log=chat_log
    )
    try:
        print("\n[BACKGROUND] Running Sleep Cycle: World State...")
        response = ollama.generate(model=MAIN_MODEL, prompt=world_prompt, format="json")
        new_state = json.loads(response['response'])
        if "player_status" in new_state and "entities" in new_state:
            update_world_state(new_state)
            print("[BACKGROUND] World state updated.")
    except Exception as e:
        print(f"[BACKGROUND] World State update failed: {e}")

    # ── Task 2: NPC Memory Update ──
    current_npc_memory = load_npc_memory()
    npc_prompt = prompts["npc_memory_prompt"].format(
        current_npc_memory=json.dumps(current_npc_memory, indent=2),
        chat_log=chat_log
    )
    try:
        print("[BACKGROUND] Running Sleep Cycle: NPC Memory...")
        response = ollama.generate(model=FAST_MODEL, prompt=npc_prompt, format="json")
        npc_update = json.loads(response['response'])
        
        if "npcs" in npc_update:
            # Merge: preserve existing NPCs not in this update
            merged = current_npc_memory.get("npcs", {})
            merged.update(npc_update["npcs"])
            save_npc_memory({"npcs": merged})
            print(f"[BACKGROUND] NPC memory updated: {list(npc_update['npcs'].keys())}")
    except Exception as e:
        print(f"[BACKGROUND] NPC Memory update failed: {e}")

    return True


# ─── Main Chat Pipeline ────────────────────────────────────────────────────────

def run_chat_turn(user_text, history):
    debug = {"router": "SKIP", "entities": [], "npcs": []}

    needs_data = router_agent(user_text)
    debug["router"] = "FETCH" if needs_data else "SKIP"

    if needs_data:
        relevant_world, relevant_npcs = librarian_agent(user_text, history)
        debug["entities"] = relevant_world
        debug["npcs"] = relevant_npcs
        targeted_context = build_targeted_context(relevant_world, relevant_npcs)
    else:
        targeted_context = build_targeted_context([], [])

    # Fine-tuned model: keep system prompt minimal — it already knows its role.
    # Only inject world context if there's actually something to inject.
    if targeted_context.strip():
        sys_content = (
            "You are the Narrator. Use this world context to inform your response "
            "but never quote it directly.\n\n"
            + targeted_context
        )
    else:
        sys_content = "You are the Narrator."

    messages = [{"role": "system", "content": sys_content}]
    messages.extend(history[-8:])

    response = generate_chat(messages)
    return response, debug