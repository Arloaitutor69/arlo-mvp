import streamlit as st
import requests

st.set_page_config(page_title="ARLO", page_icon="🌲", layout="centered")

# Custom CSS for black background + forest green theme
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

# --- Mode selection
mode = st.selectbox("Choose an action:", ["Generate Study Session", "Generate Flashcards", "Feynman Feedback"])

# --- Common inputs
topic = st.text_input("Enter topic or subject:")
notes = st.text_area("Paste notes or context here:")
submit = st.button("Submit")

if submit and topic:
    if mode == "Generate Study Session":
        response = requests.post("http://127.0.0.1:8000/generate-session", json={
            "subject": topic,
            "duration_minutes": 60,
            "notes_text": notes
        })

        st.subheader("Session Plan")
        st.json(response.json())

    elif mode == "Generate Flashcards":
        response = requests.post("http://127.0.0.1:8000/generate-flashcards", json={
            "topic": topic,
            "notes_text": notes,
            "difficulty": "medium",
            "format": "Q&A"
        })

        st.subheader("Flashcards")
        st.json(response.json())

    elif mode == "Feynman Feedback":
        response = requests.post("http://127.0.0.1:8000/feynman-feedback", json={
            "topic": topic,
            "user_explanation": notes
        })

        st.subheader("Feedback on Your Explanation")
        st.json(response.json())
