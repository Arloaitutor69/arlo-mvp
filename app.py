
import streamlit as st
import requests
import time
import json
import streamlit.components.v1 as components


# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="ARLO Tutor", layout="wide")

# ---------- CSS STYLING ----------
st.markdown("""
    <style>
        body { background-color: #000000; color: white; }
        .main { background-color: #000000; color: white; }
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
            font-size: 22px;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- SESSION STATE ----------
defaults = {
    "stage": "setup",
    "topic": "",
    "notes": "",
    "duration": 45,
    "chat_history": [],
    "current_step": 0,
    "timer_remaining": 2700,
    "timer_running": False,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ---------- PAGE HEADER ----------
st.title("🌲 ARLO — Your Personal AI Study Coach")

# ---------- SETUP SCREEN ----------
if st.session_state.stage == "setup":
    st.subheader("📋 Set Up Your Smart Study Session")

    topic = st.text_input("Topic of study", value=st.session_state.topic)
    notes = st.text_area("Paste notes here (optional)", value=st.session_state.notes)
    duration = st.slider("Total session duration (minutes)", 15, 120, 45, 5)

    if st.button("Start Session") and topic:
        st.session_state.topic = topic
        st.session_state.notes = notes
        st.session_state.duration = duration
        st.session_state.timer_remaining = duration * 60
        st.session_state.stage = "chat"
        st.session_state.chat_history = [{
            "user": "",
            "arlo": f"Hi! I'm ARLO. Let's begin our study session on **{topic}**. Ready to start?"
        }]
        st.rerun()

# ---------- CHAT SESSION ----------
elif st.session_state.stage == "chat":
    col1, col2 = st.columns([5, 1])

    with col1:
        st.subheader(f"🧠 Studying: {st.session_state.topic}")
        st.markdown("---")

        # Display conversation history
        recent_turns = st.session_state.chat_history[-6:]  # only show last 6 turns
        for turn in recent_turns:
            if turn["user"]:
                st.markdown(f"**🙋 You:** {turn['user']}")
            if turn["arlo"]:
                st.markdown(f"**🧠 ARLO:** {turn['arlo']}")



        st.markdown("---")

        # Chat input form (safe from repetition)
        with st.form("chat_input_form", clear_on_submit=True):
            user_message = st.text_input("Your response:")
            send_it = st.form_submit_button("Send")

        if send_it and user_message.strip():
            # Send to backend for ARLO reply
            payload = {
                "topic": st.session_state.topic,
                "notes_text": st.session_state.notes,
                "current_step": st.session_state.current_step,
                "user_input": user_message,
                "history": st.session_state.chat_history,
            }

            try:
                response = requests.post("http://127.0.0.1:8000/next-task", json=payload)
                reply = response.json().get("arlo_reply", "⚠️ ARLO did not reply.")
            except Exception as e:
                reply = f"⚠️ Error: {e}"

            # Append to chat history
            st.session_state.chat_history.append({
                "user": user_message,
                "arlo": reply
            })
            st.session_state.current_step += 1
            st.rerun()

    with col2:
        st.markdown("### ⏳ Time Remaining")
        mins, secs = divmod(st.session_state.timer_remaining, 60)
        st.markdown(f"### ⏳ {mins:02}:{secs:02}")
        progress = st.progress(1.0)  # full at start
        elapsed = st.session_state.duration * 60 - st.session_state.timer_remaining
        percent = max(0, 1 - (elapsed / (st.session_state.duration * 60)))
        progress.progress(percent)

        col_pause, col_add = st.columns(2)
        with col_pause:
            if st.button("⏯ Pause" if st.session_state.timer_running else "▶ Resume"):
                st.session_state.timer_running = not st.session_state.timer_running
                st.rerun()

        with col_add:
            if st.button("➕ +5 min"):
                st.session_state.timer_remaining += 300
                st.rerun()

        ###

        # Timer calculation
        elapsed = st.session_state.duration * 60 - st.session_state.timer_remaining
        percent = max(0, min(100, 100 * elapsed / (st.session_state.duration * 60)))
        remaining_minutes = st.session_state.timer_remaining // 60
        remaining_seconds = st.session_state.timer_remaining % 60
        
        circle_html = f"""
        <div style="display: flex; justify-content: center;">
          <div style="position: relative; width: 150px; height: 150px;">
            <svg viewBox="0 0 36 36" width="150" height="150">
              <path
                style="fill: none; stroke: #eee; stroke-width: 3.8;"
                d="M18 2.0845
                   a 15.9155 15.9155 0 0 1 0 31.831
                   a 15.9155 15.9155 0 0 1 0 -31.831"
              />
              <path
                style="fill: none; stroke: #00FF00; stroke-width: 3.8; stroke-dasharray: {percent}, 100; transition: stroke-dasharray 1s linear;"
                d="M18 2.0845
                   a 15.9155 15.9155 0 0 1 0 31.831
                   a 15.9155 15.9155 0 0 1 0 -31.831"
              />
              <text x="18" y="20.35" font-size="6" text-anchor="middle" fill="white">
                {remaining_minutes:02}:{remaining_seconds:02}
              </text>
            </svg>
          </div>
        </div>
        """
        
        components.html(circle_html, height=180)


        # Timer countdown (only ticks when running)
        if st.session_state.timer_running:
            time.sleep(1)
            st.session_state.timer_remaining -= 1
            if st.session_state.timer_remaining <= 0:
                st.session_state.timer_running = False
            st.rerun()
        # End session if timer hits 0
        if st.session_state.timer_remaining <= 0 and not st.session_state.timer_running:
            st.session_state.stage = "complete"
            st.rerun()

elif st.session_state.stage == "complete":
    st.success("🎉 Your session is complete!")
    st.markdown("Well done! Here's a final summary of your session and what to review tonight.")

    # Call backend for review sheet
    try:
        review_res = requests.post("http://127.0.0.1:8000/generate-review-sheet", json={
            "topic": st.session_state.topic,
            "notes_text": st.session_state.notes,
            "missed_flashcards": [],
            "feynman_feedback": ""
        })
        review = review_res.json().get("review_sheet", "No review content received.")
        st.markdown(f"### 🛏 Bedtime Review Sheet\n{review}")
    except Exception as e:
        st.error(f"⚠️ Could not fetch review sheet: {e}")

    if st.button("🔄 Start New Session"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()
