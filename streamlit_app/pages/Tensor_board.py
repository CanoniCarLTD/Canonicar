import streamlit as st

# Replace with your actual ngrok URL
tensorboard_url = "https://0201-5-29-228-0.ngrok-free.app"  # ← change this!
print(tensorboard_url)

st.title("📊 Canonicar ML Dashboard")
st.subheader("Live TensorBoard Logs")

st.components.v1.iframe(tensorboard_url, height=800)
