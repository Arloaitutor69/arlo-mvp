import streamlit as st
import requests
import time
import json

st.set_page_config(page_title="ARLO", page_icon="🌲", layout="centered")

# Styling
st.markdown("""
    <style>
        body {
            background-color: #000000;
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
    </style>
""", unsafe_allow_html=True)

st.title("🌲 ARLO — Your AI Study Partner")

# Initialize session state
if "current_task_index" not in st.session_state:
    st.session_state.current_task_index = 0
    st.session_state.in_timer = False
    st.session_state.time_per_task = 25
if "tasks" not in st.session_state:
    st.session_state.tasks = []
if "auto_mode" not in st.session_state:
    st.session_state.auto_mode = False

# Mode selection
mode = st.selectbox("Choose an action:", ["Generate Study Session", "Generate Flashcards", "Feynman Feedback", "Auto Walkthrough"])

# Common input fields
topic = st.text_input("Enter topic or subject:")
notes = st.text_area("Paste notes or context here:")

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

# Smart Study Session
if mode == "Generate Study Session":
    total_duration = st.slider(
        "⏱️ Select how long you want to study (in minutes)",
        min_value=15,
        max_value=120,
        value=45,
        step=5
    )

    if st.button("Generate Plan and Begin") and topic:
        try:
            res = requests.post("http://127.0.0.1:8000/generate-session", json={
                "subject": topic,
                "duration_minutes": total_duration,
                "notes_text": notes
            })

            if res.status_code == 200:
                raw = res.json().get("session_plan")
                session_data = json.loads(raw) if isinstance(raw, str) else raw

                if session_data:
                    tasks = session_data["tasks"]
                    num_tasks = len(tasks)
                    time_per_task = total_duration / num_tasks

                    st.session_state.tasks = tasks
                    st.session_state.current_task_index = 0
                    st.session_state.time_per_task = time_per_task
                    st.session_state.in_timer = False
                    st.session_state.auto_mode = False
                    st.rerun()
                else:
                    st.error("❌ Response received but no session plan found.")
            else:
                st.error(f"❌ Failed to generate session. Status code: {res.status_code}")
        except Exception as e:
            st.error(f"⚠️ Error while calling the backend: {e}")

# Auto Walkthrough Mode
if mode == "Auto Walkthrough" and st.session_state.tasks:
    st.session_state.auto_mode = True
    task_index = st.session_state.current_task_index
    tasks = st.session_state.tasks

    if task_index < len(tasks):
        task = tasks[task_index]
        st.subheader(f"Task {task_index + 1}")
        st.write(task)

        if not st.session_state.in_timer:
            if st.button("▶ Start Task Timer"):
                st.session_state.in_timer = True
                st.rerun()
        else:
            run_timer(int(st.session_state.time_per_task))
            st.session_state.in_timer = False
            st.session_state.current_task_index += 1
            st.rerun()
    else:
        st.success("✅ All tasks completed! Time for review.")
        st.session_state.auto_mode = False

# Flashcards
if mode == "Generate Flashcards" and st.button("Submit"):
    response = requests.post("http://127.0.0.1:8000/generate-flashcards", json={
        "topic": topic,
        "notes_text": notes,
        "difficulty": "medium",
        "format": "Q&A"
    })
    st.subheader("Flashcards")
    st.json(response.json())

# Feynman Feedback
if mode == "Feynman Feedback" and st.button("Submit"):
    response = requests.post("http://127.0.0.1:8000/feynman-feedback", json={
        "topic": topic,
        "user_explanation": notes
    })
    result = response.json().get("feynman_response")
    st.subheader("Feedback")
    st.write(result["feedback"])
    st.subheader("Follow-Up Questions")
    for q in result["follow_up_questions"]:
        st.markdown(f"• {q}")
