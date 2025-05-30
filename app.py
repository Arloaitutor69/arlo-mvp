# ARLO Final app.py (Full MVP) — Setup screen, session flow, timer, chatbot
import streamlit as st
import requests
import time
import json
import tempfile
import os
from pyvis.network import Network
import streamlit.components.v1 as components

st.set_page_config(page_title="ARLO Study Session", layout="wide")

# --- Style ---
st.markdown("""
    <style>
        body { background-color: #000000; }
        .main { background-color: #000000; color: white; }
        header, .css-18e3th9 { background-color: #014421 !important; color: white; }
        .stButton>button { background-color: #014421; color: white; }
        .timer-circle {
            border-radius: 50%;
            border: 6px solid #00cc66;
            padding: 10px;
            text-align: center;
            width: 120px;
            height: 120px;
            display: flex;
            justify-content: center;
            align-items: center;
            font-size: 24px;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# --- Init State ---
defaults = {
    "stage": "setup", "topic": "", "notes": "",
    "current_task": 0, "timer_remaining": 0,
    "timer_running": False, "tasks": [], "time_per_task": 25
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.title("🌲 ARLO — Personalized AI Study Partner")

# --- Setup Phase ---
if st.session_state.stage == "setup":
    st.header("📋 Setup Your Study Session")
    topic = st.text_input("What topic are you studying?")
    notes = st.text_area("Paste any notes or context here:")
    duration = st.slider("How many minutes would you like to study?", 15, 120, 45, 5)

    if st.button("Start Smart Study Session") and topic:
        try:
            res = requests.post("http://127.0.0.1:8000/generate-session", json={
                "subject": topic,
                "duration_minutes": duration,
                "notes_text": notes
            })
            if res.status_code == 200:
                raw = res.json().get("session_plan")
                plan = json.loads(raw) if isinstance(raw, str) else raw
                st.session_state.tasks = plan["tasks"]
                st.session_state.time_per_task = duration / max(len(plan["tasks"]), 1)
                st.session_state.current_task = 0
                st.session_state.timer_remaining = int(st.session_state.time_per_task * 60)
                st.session_state.timer_running = False
                st.session_state.topic = topic
                st.session_state.notes = notes
                st.session_state.stage = "session"
                st.rerun()
            else:
                st.error("Could not generate study session.")
        except Exception as e:
            st.error(f"Error generating session: {e}")

# --- Session Screen ---
elif st.session_state.stage == "session":
    st.header(f"🧠 Studying: {st.session_state.topic}")
    col1, col2 = st.columns([5, 1])

    with col1:
        tasks = st.session_state.tasks
        idx = st.session_state.current_task
        if idx < len(tasks):
            task = tasks[idx]
            st.subheader(f"Task {idx + 1}/{len(tasks)}")
            st.markdown(task)

            if isinstance(task, str) and "flashcard" in task.lower():
                res = requests.post("http://127.0.0.1:8000/generate-flashcards", json={
                    "topic": st.session_state.topic, "notes_text": st.session_state.notes,
                    "difficulty": "medium", "format": "Q&A"
                })
                st.json(res.json())

            elif isinstance(task, str) and "feynman" in task.lower():
                exp = st.text_area("Explain the topic in your own words:")
                if st.button("Submit Explanation"):
                    feyn = requests.post("http://127.0.0.1:8000/feynman-feedback", json={
                        "topic": st.session_state.topic, "user_explanation": exp
                    })
                    parsed = json.loads(feyn.json().get("feynman_response", "{}"))
                    st.write(parsed.get("feedback", ""))
                    for q in parsed.get("follow_up_questions", []):
                        st.markdown(f"- {q}")

            elif isinstance(task, str) and "blurting" in task.lower():
                blurt = st.text_area("Write everything you know about the topic:")
                if st.button("Submit Blurting"):
                    res = requests.post("http://127.0.0.1:8000/blurting-feedback", json={
                        "topic": st.session_state.topic,
                        "user_blurting": blurt,
                        "reference_notes": st.session_state.notes
                    })
                    st.markdown(res.json().get("blurting_feedback", ""))

            elif isinstance(task, str) and "mind map" in task.lower():
                res = requests.post("http://127.0.0.1:8000/generate-mindmap", json={
                    "topic": st.session_state.topic,
                    "notes_text": st.session_state.notes
                })
                data = json.loads(res.json().get("mindmap", "{}"))
                net = Network(height="500px", width="100%", bgcolor="#000000", font_color="white")
                central = data.get("Central Idea", st.session_state.topic)
                net.add_node(central, label=central, color="#00cc66")
                for branch, subs in data.get("Branches", {}).items():
                    net.add_node(branch, label=branch, color="#1f7a1f")
                    net.add_edge(central, branch)
                    for sub in subs:
                        net.add_node(sub, label=sub, color="#004d26")
                        net.add_edge(branch, sub)
                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
                net.save_graph(tmp_file.name)
                components.html(open(tmp_file.name).read(), height=550)
                os.unlink(tmp_file.name)

            if st.button("⏭ Skip to Next Task"):
                st.session_state.current_task += 1
                st.session_state.timer_remaining = int(st.session_state.time_per_task * 60)
                st.rerun()
        else:
            st.success("🎉 All tasks complete!")
            review = requests.post("http://127.0.0.1:8000/generate-review-sheet", json={
                "topic": st.session_state.topic,
                "notes_text": st.session_state.notes,
                "missed_flashcards": [],
                "feynman_feedback": ""
            })
            st.subheader("🛏 Bedtime Review Sheet")
            st.markdown(review.json().get("review_sheet", ""))
            st.session_state.stage = "complete"

    with col2:
        st.markdown("### ⏳ Time Left")
        mins, secs = divmod(st.session_state.timer_remaining, 60)
        st.markdown(f"<div class='timer-circle'>{mins:02}:{secs:02}</div>", unsafe_allow_html=True)
        if st.button("⏸ Pause" if st.session_state.timer_running else "▶ Resume"):
            st.session_state.timer_running = not st.session_state.timer_running
        if st.button("+5 min"):
            st.session_state.timer_remaining += 5 * 60
        if st.session_state.timer_running:
            time.sleep(1)
            st.session_state.timer_remaining -= 1
            st.rerun()

# --- ARLO Chat Sidebar ---
with st.sidebar:
    st.header("🤖 Chat with ARLO")
    user_question = st.text_input("Ask ARLO anything:")
    if user_question:
        chat_res = requests.post("http://127.0.0.1:8000/arlo-chat", json={
            "topic": st.session_state.topic,
            "notes_text": st.session_state.notes,
            "question": user_question
        })
        st.markdown("**ARLO says:**")
        st.markdown(chat_res.json().get("reply", "No response."))

