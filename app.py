# app.py — Flask + React SPA
from flask import Flask, request, jsonify, Response
import threading
from config import SUMMARY_THRESHOLD
from memory_manager import load_history, save_json, HISTORY_FILE, append_to_history
from world_state import load_world_state, update_world_state, get_state_summary
from agent_pipeline import (
    run_chat_turn, run_sleep_cycle,
    load_prompts, save_prompts,
    load_npc_memory,
)

app = Flask(__name__)
HTML = open("ui.html", encoding="utf-8").read()

@app.route('/')
def home():
    return Response(HTML, mimetype='text/html')

@app.route('/api/history', methods=['GET'])
def get_history():
    return jsonify({"history": load_history()})

@app.route('/api/history/edit', methods=['POST'])
def edit_history():
    data = request.json
    idx, new_content = data.get("index"), data.get("content")
    history = load_history()
    if 0 <= idx < len(history):
        history[idx]["content"] = new_content
        save_json(HISTORY_FILE, history)
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "Index out of range"}), 400

@app.route('/api/history/delete', methods=['POST'])
def delete_history_msg():
    idx = request.json.get("index")
    history = load_history()
    if idx is not None and 0 <= idx < len(history):
        history.pop(idx)
        save_json(HISTORY_FILE, history)
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "Index out of range"}), 400

@app.route('/api/history/clear', methods=['POST'])
def clear_history():
    save_json(HISTORY_FILE, [])
    return jsonify({"success": True})

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    return jsonify({"summary": get_state_summary()})

@app.route('/api/prompts', methods=['GET', 'POST'])
def handle_prompts():
    if request.method == 'POST':
        save_prompts(request.json)
        return jsonify({"success": True})
    return jsonify(load_prompts())

@app.route('/api/world', methods=['GET', 'POST'])
def handle_world():
    if request.method == 'POST':
        update_world_state(request.json)
        return jsonify({"success": True})
    return jsonify(load_world_state())

@app.route('/api/npcs', methods=['GET'])
def get_npcs():
    return jsonify(load_npc_memory())

@app.route('/api/sleep', methods=['POST'])
def force_sleep():
    history = load_history()
    threading.Thread(target=run_sleep_cycle, args=(history,), daemon=True).start()
    return jsonify({"success": True})

@app.route('/api/chat', methods=['POST'])
def handle_chat_turn():
    user_text = request.json.get("message", "").strip()
    if not user_text:
        return jsonify({"error": "Empty input"}), 400
    append_to_history("user", user_text)
    history = load_history()
    try:
        response, debug = run_chat_turn(user_text, history)
    except Exception as e:
        response = f"[Error: {e}]"
        debug = {"router": "ERROR", "entities": [], "npcs": [], "tone": "ERROR",
                 "tokens": {"prompt": 0, "response": 0, "total": 0, "limit": 8192, "pct": 0}}
    append_to_history("assistant", response)
    if len(history) % SUMMARY_THRESHOLD == 0:
        threading.Thread(target=run_sleep_cycle, args=(history,), daemon=True).start()
    return jsonify({"response": response, "debug": debug})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)