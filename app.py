
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
            
                        
                # ---------- TOOL EXECUTION RENDERING ----------                
                if "tool_mode" not in st.session_state:
                    st.warning("Select a tool mode.")
                elif st.session_state.tool_mode == "flashcard":
                    st.subheader("📚 Flashcard Practice")
            
                    cards = st.session_state.get("flashcards", [])
                    index = st.session_state.get("flash_index", 0)
            
                    if index < len(cards):
                        card = cards[index]
                        question = card['question']
                        answer = card['answer']
            
                        import streamlit.components.v1 as components
            
                        html_card = f"""
                        <style>
                            .flip-card {{
                              background-color: transparent;
                              width: 300px;
                              height: 200px;
                              perspective: 1000px;
                              margin: auto;
                            }}
                            .flip-card-inner {{
                              position: relative;
                              width: 100%;
                              height: 100%;
                              text-align: center;
                              transition: transform 0.6s;
                              transform-style: preserve-3d;
                            }}
                            .flip-card:hover .flip-card-inner {{
                              transform: rotateY(180deg);
                            }}
                            .flip-card-front, .flip-card-back {{
                              position: absolute;
                              width: 100%;
                              height: 100%;
                              -webkit-backface-visibility: hidden;
                              backface-visibility: hidden;
                              display: flex;
                              align-items: center;
                              justify-content: center;
                              border: 2px solid #014421;
                              border-radius: 10px;
                              font-size: 18px;
                              padding: 10px;
                            }}
                            .flip-card-front {{
                              background-color: #014421;
                              color: white;
                            }}
                            .flip-card-back {{
                              background-color: #ffffff;
                              color: black;
                              transform: rotateY(180deg);
                            }}
                        </style>
                        <div class="flip-card">
                          <div class="flip-card-inner">
                            <div class="flip-card-front">
                              {question}
                            </div>
                            <div class="flip-card-back">
                              {answer}
                            </div>
                          </div>
                        </div>
                        """
            
                        components.html(html_card, height=250)
            
                        if st.button("Next Flashcard"):
                            st.session_state.flash_index += 1
                            st.rerun()
                    else:
                        st.success("✅ You've completed all flashcards.")
                        st.session_state.tool_mode = None

                elif st.session_state.tool_mode == "blurting":
                    st.subheader("🗣 Blurting Practice")
                    user_blurt = st.text_area("Write everything you remember (from memory):")
                    if st.button("Submit Blurting"):
                        res = requests.post("http://127.0.0.1:8000/blurting-feedback", json={
                            "topic": st.session_state.topic,
                            "user_blurting": user_blurt,
                            "reference_notes": st.session_state.notes
                        })
                        feedback = res.json().get("blurting_feedback", "")
                        st.markdown("### 🔍 Feedback")
                        st.markdown(feedback)
                        st.session_state.tool_mode = None
                        st.rerun()
            
                elif st.session_state.tool_mode == "feynman":
                    st.subheader("🧠 Feynman Technique (Teach Back)")
                    explanation = st.text_area("Explain the concept in your own words:")
                    if st.button("Submit Explanation"):
                        res = requests.post("http://127.0.0.1:8000/feynman-feedback", json={
                            "topic": st.session_state.topic,
                            "user_explanation": explanation
                        })
                        result = res.json().get("feynman_response", {})
                        st.markdown("### 💬 Feedback")
                        st.markdown(result.get("feedback", "No feedback."))
                        st.markdown("### Follow-Up Questions")
                        for q in result.get("follow_up_questions", []):
                            st.markdown(f"- {q}")
                        st.session_state.tool_mode = None
                        st.rerun()

            
                elif st.session_state.tool_mode == "mindmap":
                    st.subheader("🧭 Interactive Mind Map")
                    try:
                        res = requests.post("http://127.0.0.1:8000/generate-mindmap", json={
                            "topic": st.session_state.topic,
                            "notes_text": st.session_state.notes
                        })
                        data = res.json()
                        topic = data.get("topic", "Mind Map")
                        nodes = data.get("nodes", {})
            
                        from pyvis.network import Network
                        import networkx as nx
                        import streamlit.components.v1 as components
            
                        net = Network(height="600px", width="100%", bgcolor="#000000", font_color="white")
                        net.barnes_hut()
            
                        net.add_node(topic, label=topic, shape='ellipse', color="#00FF00")
            
                        for sub, children in nodes.items():
                            net.add_node(sub, label=sub, shape='box', color="#228B22")
                            net.add_edge(topic, sub)
                            for child in children:
                                net.add_node(child, label=child, shape='dot', color="#AAAAAA")
                                net.add_edge(sub, child)
            
                        net.save_graph("/tmp/mindmap.html")
                        HtmlFile = open("/tmp/mindmap.html", 'r', encoding='utf-8')
                        source_code = HtmlFile.read()
                        components.html(source_code, height=600, scrolling=True)
            
                    except Exception as e:
                        st.error(f"⚠️ Could not load mind map: {e}")
            
                    if st.button("✅ Done with Mind Map"):
                        st.session_state.tool_mode = None
                        st.rerun()


            # Intelligent response handling
            insert_tool = None
            tool_data = None
            
            # Check what ARLO said
            if isinstance(reply, str):
                reply_lower = reply.lower()
                if "flashcard" in reply_lower:
                    insert_tool = "flashcard"
                elif "blurting" in reply_lower:
                    insert_tool = "blurting"
                elif "feynman" in reply_lower or "teach back" in reply_lower:
                    insert_tool = "feynman"
                elif "mind map" in reply_lower:
                    insert_tool = "mindmap"
            
            # Save ARLO message
            st.session_state.chat_history.append({
                "user": user_message,
                "arlo": reply
            })
            st.session_state.current_step += 1
            
            # Trigger study tool interaction
            if insert_tool == "flashcard":
                st.session_state.tool_mode = "flashcard"
                res = requests.post("http://127.0.0.1:8000/generate-flashcards", json={
                    "topic": st.session_state.topic,
                    "notes_text": st.session_state.notes,
                    "difficulty": "medium",
                    "format": "Q&A"
                })
                flashcards = res.json().get("flashcards", [])
                st.session_state.flashcards = flashcards
                st.session_state.flash_index = 0
                st.rerun()
            
            elif insert_tool == "blurting":
                st.session_state.tool_mode = "blurting"
                st.rerun()
            
            elif insert_tool == "feynman":
                st.session_state.tool_mode = "feynman"
                st.rerun()
            
            elif insert_tool == "mindmap":
                st.session_state.tool_mode = "mindmap"
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
