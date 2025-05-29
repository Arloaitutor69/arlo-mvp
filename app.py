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

mode = st.selectbox("Choose an action:", [
    "Generate Study Session", 
    "Generate Flashcards", 
    "Feynman Feedback", 
    "Blurting Practice", 
    "Auto Walkthrough"
])

# Initialize session state
if "current_task_index" not in st.session_state:
    st.session_state.current_task_index = 0
    st.session_state.in_timer = False
    st.session_state.time_per_task = 25
if "tasks" not in st.session_state:
    st.session_state.tasks = []
if "auto_mode" not in st.session_state:
    st.session_state.auto_mode = False
if "session_started" not in st.session_state:
    st.session_state.session_started = False
if "last_flashcards" not in st.session_state:
    st.session_state.last_flashcards = {}
if "feynman_result" not in st.session_state:
    st.session_state.feynman_result = {}

# Common input fields
topic = st.text_input("Enter topic or subject:")
notes = st.text_area("Paste notes or context here:")
total_duration = st.slider(
    "⏱️ Select how long you want to study (in minutes)",
    min_value=15,
    max_value=120,
    value=45,
    step=5
)

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

            if session_data:
                tasks = session_data["tasks"]
                num_tasks = len(tasks)
                time_per_task = total_duration / num_tasks

                st.session_state.tasks = tasks
                st.session_state.current_task_index = 0
                st.session_state.time_per_task = time_per_task
                st.session_state.in_timer = False
                st.session_state.auto_mode = True
                st.session_state.session_started = True
                st.rerun()
            else:
                st.error("❌ Response received but no session plan found.")
        else:
            st.error(f"❌ Failed to generate session. Status code: {res.status_code}")
    except Exception as e:
        st.error(f"⚠️ Error while calling the backend: {e}")

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

# Auto Walkthrough Activated Automatically After Session Starts
if st.session_state.auto_mode and st.session_state.tasks:
    task_index = st.session_state.current_task_index
    tasks = st.session_state.tasks

    if task_index < len(tasks):
        task = tasks[task_index]
        st.subheader(f"Task {task_index + 1}")
        st.write(task)

        # Trigger dynamic tool use
        if isinstance(task, str) and "flashcard" in task.lower():
            st.info("📖 ARLO is generating flashcards for this step...")
            flash_res = requests.post("http://127.0.0.1:8000/generate-flashcards", json={
                "topic": topic,
                "notes_text": notes,
                "difficulty": "medium",
                "format": "Q&A"
            })
            flashcards = flash_res.json()
            st.session_state.last_flashcards = flashcards
            st.json(flashcards)

        elif isinstance(task, str) and "feynman" in task.lower():
            st.info("🎤 ARLO is evaluating your explanation...")
            user_exp = st.text_area("Write your explanation in your own words:")
            if st.button("Submit Feynman Explanation"):
                feyn_res = requests.post("http://127.0.0.1:8000/feynman-feedback", json={
                    "topic": topic,
                    "user_explanation": user_exp
                })
                feedback_raw = feyn_res.json().get("feynman_response")
                parsed = json.loads(feedback_raw) if isinstance(feedback_raw, str) else feedback_raw
                st.session_state.feynman_result = parsed
                st.subheader("Feedback")
                st.write(parsed["feedback"])
                st.subheader("Follow-Up Questions")
                for q in parsed["follow_up_questions"]:
                    st.markdown(f"- {q}")

        if not st.session_state.in_timer:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("▶ Start Task Timer"):
                    st.session_state.in_timer = True
                    st.rerun()
            with col2:
                if st.button("⏭ I finished this task"):
                    st.session_state.current_task_index += 1
                    st.rerun()
        else:
            run_timer(int(st.session_state.time_per_task))
            st.session_state.in_timer = False
            st.session_state.current_task_index += 1
            st.rerun()
    else:
        st.success("✅ All tasks completed! Time for review.")

        missed = []
        if "last_flashcards" in st.session_state:
            missed = st.session_state.last_flashcards.get("flashcards", [])[-3:]
        feyn = ""
        if "feynman_result" in st.session_state:
            feyn = st.session_state.feynman_result.get("feedback", "")
        review_res = requests.post("http://127.0.0.1:8000/generate-review-sheet", json={
            "topic": topic,
            "notes_text": notes,
            "missed_flashcards": missed,
            "feynman_feedback": feyn
        })
        review = review_res.json().get("review_sheet")
        st.subheader("🛎 Bedtime Review Sheet")
        st.markdown(review)
        st.session_state.auto_mode = False

# Flashcards (standalone)
if mode == "Generate Flashcards" and st.button("Submit"):
    response = requests.post("http://127.0.0.1:8000/generate-flashcards", json={
        "topic": topic,
        "notes_text": notes,
        "difficulty": "medium",
        "format": "Q&A"
    })
    st.subheader("Flashcards")
    st.json(response.json())

# Feynman Feedback (standalone)
if mode == "Feynman Feedback" and st.button("Submit"):
    response = requests.post("http://127.0.0.1:8000/feynman-feedback", json={
        "topic": topic,
        "user_explanation": notes
    })
    result = response.json().get("feynman_response")
    parsed = json.loads(result) if isinstance(result, str) else result
    st.subheader("Feedback")
    st.write(parsed["feedback"])
    st.subheader("Follow-Up Questions")
    for q in parsed["follow_up_questions"]:
        st.markdown(f"- {q}")

# Blurting Practice (standalone)
if mode == "Blurting Practice":
    user_blurt = st.text_area("Blurt out everything you know (from memory):")
    if st.button("Submit Blurting"):
        if user_blurt:
            response = requests.post("http://127.0.0.1:8000/blurting-feedback", json={
                "topic": topic,
                "user_blurting": user_blurt,
                "reference_notes": notes
            })
            feedback = response.json().get("blurting_feedback", "")
            st.subheader("Feedback — What You Missed")
            st.markdown(feedback)
