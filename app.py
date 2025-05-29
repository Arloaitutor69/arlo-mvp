
# === FINALIZED app.py with Smart Walkthrough, Manual Tool Launch, and ARLO Chatbot ===
import streamlit as st
import requests
import time
import json
import tempfile
import os
from pyvis.network import Network
import streamlit.components.v1 as components

st.set_page_config(page_title="ARLO", page_icon="🌲", layout="centered")

# Styling
st.markdown("""
    <style>
        body { background-color: #000000; }
        .main { background-color: #000000; color: white; }
        header, .css-18e3th9 {
            background-color: #014421 !important;
            color: white;
        }
        .stButton>button {
            background-color: #014421;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🌲 ARLO — Your AI Study Partner")

mode = st.selectbox("Choose an action:", [
    "Generate Study Session", 
    "Auto Walkthrough",
    "Run a Study Technique",
    "ARLO Chatbot"
])

# Initialize session state
for key, default in {
    "current_task_index": 0,
    "in_timer": False,
    "time_per_task": 25,
    "tasks": [],
    "auto_mode": False,
    "session_started": False,
    "last_flashcards": {},
    "feynman_result": {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

topic = st.text_input("Enter topic or subject:")
notes = st.text_area("Paste notes or context here:")
total_duration = st.slider("⏱️ Study Duration (minutes)", 15, 120, 45, 5)

# Session Planner
if mode == "Generate Study Session":
    if st.button("Start My Smart Session") and topic:
        try:
            res = requests.post("http://127.0.0.1:8000/generate-session", json={
                "subject": topic,
                "duration_minutes": total_duration,
                "notes_text": notes
            })
            if res.status_code == 200:
                raw = res.json().get("session_plan")
                session_data = json.loads(raw) if isinstance(raw, str) else raw
                tasks = session_data.get("tasks", [])
                st.session_state.tasks = tasks
                st.session_state.time_per_task = total_duration / max(len(tasks), 1)
                st.session_state.current_task_index = 0
                st.session_state.auto_mode = True
                st.rerun()
        except Exception as e:
            st.error(f"❌ Failed: {e}")

# Timer function
def run_timer(minutes):
    total_seconds = minutes * 60
    progress = st.empty()
    timer_text = st.empty()
    for i in range(total_seconds, 0, -1):
        mins, secs = divmod(i, 60)
        timer_text.markdown(f"### ⏳ Time Remaining: {mins:02d}:{secs:02d}")
        progress.progress((total_seconds - i) / total_seconds)
        time.sleep(1)
    timer_text.markdown("### ✅ Time’s up!")
    progress.empty()

# Keyword Detection
def detect_task_type(task):
    task = task.lower()
    if any(kw in task for kw in ["flashcard", "review", "quiz"]):
        return "flashcard"
    if any(kw in task for kw in ["feynman", "teach", "explain"]):
        return "feynman"
    if any(kw in task for kw in ["blurt", "recall", "dump"]):
        return "blurting"
    if any(kw in task for kw in ["mind map", "diagram", "concept"]):
        return "mindmap"
    return "text"

# Walkthrough Engine
if mode == "Auto Walkthrough" and st.session_state.auto_mode:
    idx = st.session_state.current_task_index
    tasks = st.session_state.tasks

    if idx < len(tasks):
        task = tasks[idx]
        st.subheader(f"Task {idx + 1}")
        st.write(task)
        tool = detect_task_type(task)

        if tool == "flashcard":
            st.info("📖 ARLO is generating flashcards...")
            res = requests.post("http://127.0.0.1:8000/generate-flashcards", json={
                "topic": topic, "notes_text": notes, "difficulty": "medium", "format": "Q&A"
            })
            flashcards = res.json()
            st.session_state.last_flashcards = flashcards
            st.json(flashcards)

        elif tool == "feynman":
            user_exp = st.text_area("Explain this in your own words:")
            if st.button("Submit Feynman"):
                feyn_res = requests.post("http://127.0.0.1:8000/feynman-feedback", json={
                    "topic": topic, "user_explanation": user_exp
                })
                raw = feyn_res.json().get("feynman_response", "{}")
                parsed = json.loads(raw) if isinstance(raw, str) else raw
                st.session_state.feynman_result = parsed
                st.write(parsed.get("feedback", ""))
                for q in parsed.get("follow_up_questions", []):
                    st.markdown(f"- {q}")

        elif tool == "blurting":
            blurt = st.text_area("Blurt out everything you remember:")
            if st.button("Submit Blurting"):
                res = requests.post("http://127.0.0.1:8000/blurting-feedback", json={
                    "topic": topic, "user_blurting": blurt, "reference_notes": notes
                })
                st.markdown(res.json().get("blurting_feedback", ""))

        elif tool == "mindmap":
            st.info("🧠 Building mind map...")
            res = requests.post("http://127.0.0.1:8000/generate-mindmap", json={"topic": topic, "notes_text": notes})
            parsed = json.loads(res.json().get("mindmap", "{}"))
            net = Network(height="600px", width="100%", bgcolor="#000000", font_color="white")
            central = parsed.get("Central Idea", topic)
            net.add_node(central, label=central, color="#00cc66")
            for branch, subs in parsed.get("Branches", {}).items():
                net.add_node(branch, label=branch, color="#1f7a1f")
                net.add_edge(central, branch)
                for sub in subs:
                    net.add_node(sub, label=sub, color="#004d26")
                    net.add_edge(branch, sub)
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
            net.save_graph(tmp_file.name)
            components.html(open(tmp_file.name, "r").read(), height=650)
            os.unlink(tmp_file.name)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶ Start Timer"):
                st.session_state.in_timer = True
                st.rerun()
        with col2:
            if st.button("⏭ Next Task"):
                st.session_state.current_task_index += 1
                st.session_state.in_timer = False
                st.rerun()

        if st.session_state.in_timer:
            run_timer(int(st.session_state.time_per_task))
            st.session_state.in_timer = False
            st.session_state.current_task_index += 1
            st.rerun()
    else:
        st.success("✅ All tasks complete!")
        missed = st.session_state.last_flashcards.get("flashcards", [])[-3:]
        feyn = st.session_state.feynman_result.get("feedback", "")
        res = requests.post("http://127.0.0.1:8000/generate-review-sheet", json={
            "topic": topic, "notes_text": notes, "missed_flashcards": missed, "feynman_feedback": feyn
        })
        st.markdown(res.json().get("review_sheet", ""))
        st.session_state.auto_mode = False

# === Manual Tool Runner ===
if mode == "Run a Study Technique":
    choice = st.radio("Pick a technique to run now:", ["Flashcards", "Feynman", "Blurting", "Mind Map"])
    if st.button("Run Selected Tool"):
        st.session_state.auto_mode = False
        st.session_state.tasks = [choice.lower()]
        st.session_state.current_task_index = 0
        st.rerun()

# === ARLO Chatbot ===
if mode == "ARLO Chatbot":
    st.markdown("Chat with ARLO about your topic:")
    chat = st.chat_input("Ask ARLO a question...")
    if chat:
        res = requests.post("http://127.0.0.1:8000/arlo-chat", json={
            "topic": topic,
            "notes_text": notes,
            "question": chat
        })
        st.write(res.json().get("reply", "⚠️ No response."))
