
import streamlit as st
import requests
import time
import json

st.set_page_config(page_title="ARLO Tutor", page_icon="🌱", layout="wide")

# ---------- STYLING ----------
st.markdown("""
    <style>
        body {
            background-color: #000000;
            color: white;
        }
        .main {
            background-color: #000000;
            color: white;
        }
        header, .css-18e3th9 {
            background-color: #014421 !important;
            color: white;
        }
        .stButton>button {
            background-color: #014421;
            color: white;
        }
        .timer-box {
            border: 2px solid #014421;
            padding: 10px;
            border-radius: 10px;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- STATE INIT ----------
if "stage" not in st.session_state:
    st.session_state.stage = "setup"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_step" not in st.session_state:
    st.session_state.current_step = 0
if "topic" not in st.session_state:
    st.session_state.topic = ""
if "notes" not in st.session_state:
    st.session_state.notes = ""
if "duration" not in st.session_state:
    st.session_state.duration = 45
if "timer_running" not in st.session_state:
    st.session_state.timer_running = False
if "timer_remaining" not in st.session_state:
    st.session_state.timer_remaining = 60 * 45

# ---------- PAGE LOGIC ----------
st.title("🌱 ARLO — Your AI Study Partner")

if st.session_state.stage == "setup":
    st.subheader("Build Your Study Session")

    topic = st.text_input("What are you studying today?")
    notes = st.text_area("Paste your notes or leave blank:")
    duration = st.slider("Study session length (minutes)", 15, 120, 45, 5)

    if st.button("Start My Smart Session"):
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

                # Prime ARLO chat
                arlo_intro = f"Hi! I'm ARLO, your study coach. Let's begin your session on **{topic}**. Ready?"
                st.session_state.chat_history.append({
                    "user": "",
                    "arlo": arlo_intro
                })
                st.rerun()
            else:
                st.error("❌ Could not generate session.")
        except Exception as e:
            st.error(f"⚠️ Error: {e}")

elif st.session_state.stage == "session":
    col1, col2 = st.columns([4, 1])

    with col1:
        st.subheader("🧠 ARLO Tutor Session")

        # Display chat history
        for msg in st.session_state.chat_history:
            if msg["arlo"]:
                st.markdown(f"**🧠 ARLO:** {msg['arlo']}")
            if msg["user"]:
                st.markdown(f"**🙋 You:** {msg['user']}")

        st.markdown("---")
        user_input = st.text_input("Type your response and press Enter:", key="input_box")

        if user_input:
            # Clear input on submit
            st.session_state.input_box = ""  # 👈 this resets the input box

        if user_input:
            # Send to /next-task
            payload = {
                "topic": st.session_state.topic,
                "notes_text": st.session_state.notes,
                "current_step": st.session_state.current_step,
                "user_input": user_input,
                "history": st.session_state.chat_history
            }

            try:
                res = requests.post("http://127.0.0.1:8000/next-task", json=payload)
                arlo_reply = res.json().get("arlo_reply", "⚠️ ARLO did not reply.")
            except Exception as e:
                arlo_reply = f"⚠️ Error contacting ARLO: {e}"

            # Update session state
            st.session_state.chat_history.append({
                "user": user_input,
                "arlo": arlo_reply
            })
            st.session_state.current_step += 1
            st.rerun()

    with col2:
        st.markdown("### ⏳ Timer")
        mins, secs = divmod(st.session_state.timer_remaining, 60)
        st.markdown(f"<div class='timer-box'><h2>{mins:02}:{secs:02}</h2></div>", unsafe_allow_html=True)

        if st.button("⏸ Pause" if st.session_state.timer_running else "▶ Resume"):
            st.session_state.timer_running = not st.session_state.timer_running

        if st.button("➕ Add 5 Min"):
            st.session_state.timer_remaining += 5 * 60

        if st.session_state.timer_running:
            time.sleep(1)
            st.session_state.timer_remaining -= 1
            st.rerun()
