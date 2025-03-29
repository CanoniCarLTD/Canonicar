import streamlit as st

# Custom CSS for full-screen video and centered title
st.markdown("""
    <style>
        .title {
            text-align: center;
            font-size: 40px;
            font-weight: bold;
        }
        .video-container {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 80vh; 
        }
        iframe {
            width: 90vw; 
            height: 80vh; 
            border-radius: 10px; 
        }
    </style>
""", unsafe_allow_html=True)

# Centered title
st.markdown('<div class="title">Live YouTube Video</div>', unsafe_allow_html=True)

video_id = "DFOeRcof5Zs" 
live_embed = f"""
<div class="video-container">
    <iframe src="https://www.youtube.com/embed/{video_id}?autoplay=1&controls=0&modestbranding=1&rel=0&vq=hd1080" 
    frameborder="0" allow="autoplay; encrypted-media; fullscreen" allowfullscreen></iframe>
</div>
"""
st.markdown(live_embed, unsafe_allow_html=True)
