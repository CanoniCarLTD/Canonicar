import streamlit as st

# Replace with your actual ngrok URL
tensorboard_url = "https://2f48-5-29-145-151.ngrok-free.app"  # ← change this!

st.title("📊 Canonicar ML Dashboard")
st.subheader("Live TensorBoard Logs")

st.components.v1.iframe(tensorboard_url, height=800)