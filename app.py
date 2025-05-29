import streamlit as st
import requests
import time

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

# Mode selection
mode = st.selectbox("Choose an action:", ["Generate Study Session", "Generate Flashcards", "Feynman Feedback"])

# Common input fields
topic = st.text_input("Enter topic or subject:")
notes = st.text_area("Paste notes or context here:")

def run_timer(minutes):
    start_time = time.time()
    total_seconds = minutes * 60
    while time.time() - start_time < total_seconds:
        elapsed = int(time.time() - start_time)
        remaining = total_seconds - elapsed
        mins, secs = divmod(remaining, 60)
        st.markdown(f"### ⏳ Time Remaining: {int(mins):02}:{int(secs):02}")
        time.sleep(1)

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
        res = requests.post("http://127.0.0.1:8000/generate-session", json={
            "subject": topic,
            "duration_minutes": total_duration,
            "notes_text": notes
        })

        if res.status_code == 200:
    try:
        session_data = res.json().get("session_plan")
        if session_data:
            tasks = session_data["tasks"]
            num_tasks = len(tasks)
            time_per_task = total_duration / num_tasks

            st.session_state.tasks = tasks
            st.session_state.current = 0
            st.session_state.time_per_task = time_per_task
            st.experimental_rerun()
        else:
            st.error("❌ Response received but no session plan found.")
    except Exception as e:
        st.error(f"⚠️ Error parsing response: {e}")
else:
    st.error(f"❌ Failed to generate session. Server returned status {res.status_code}")


# Run Pomodoro task loop
if "tasks" in st.session_state and st.session_state.current < len(st.session_state.tasks):
    current_task = st.session_state.tasks[st.session_state.current]
    st.markdown(f"## 📚 Task {st.session_state.current + 1}:")
    st.write(current_task)

    extend = st.button("🕒 +5 minutes")
    skip = st.button("⏭ Move to next task")
    run = st.button("▶ Start timer")

    if run:
        run_timer(st.session_state.time_per_task)

    if extend:
        st.session_state.time_per_task += 5
        st.experimental_rerun()

    if skip:
        st.session_state.current += 1
        st.session_state.time_per_task = total_duration / len(st.session_state.tasks)
        st.experimental_rerun()

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
