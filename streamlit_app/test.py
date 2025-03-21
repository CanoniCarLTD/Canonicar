import streamlit as st
import websocket
import json
import threading
from streamlit.runtime.scriptrunner import add_script_run_ctx

# Set up Foxglove connection details
foxglove_ip = "5.29.228.0"
foxglove_port = "8765"
foxglove_url = f"ws://{foxglove_ip}:{foxglove_port}"

# Initialize session state to store messages by topic
if 'messages_by_topic' not in st.session_state:
    st.session_state.messages_by_topic = {}

# Create a placeholder for the connection status
status_placeholder = st.empty()

# Topic filter
topic_filter = st.text_input("Filter by topic (leave empty for all topics)", "")


def on_message(_, message):
    """Extracts relevant Foxglove WebSocket messages."""
    try:
        data = json.loads(message)

        # Debug: Print raw message for debugging
        print(f"Full received message: {data}")

        # Extract the topic name (using 'name' instead of 'topic')
        topic = data.get("name", "unknown_topic")

        # Extract relevant content (if 'data' key exists, use it; otherwise, take full message)
        relevant_data = data.get("responseSchema", "No Relevant Data")

        # Debug: Print topic and message
        print(f"\nTopic: {topic}\nMessage: {relevant_data}")

        # Store message in session state
        if topic not in st.session_state.messages_by_topic:
            st.session_state.messages_by_topic[topic] = []

        st.session_state.messages_by_topic[topic] = (
            st.session_state.messages_by_topic[topic][-9:] + [relevant_data]
        )

    except json.JSONDecodeError:
        st.error("Received non-JSON message")
    except Exception as e:
        st.error(f"Error processing message: {str(e)}")


def on_error(_, error):
    status_placeholder.error(f"WebSocket error: {error}")


def on_close(_, *args):
    status_placeholder.warning("WebSocket connection closed")


def on_open(ws):
    """Called when the WebSocket connection is opened."""
    status_placeholder.success("Connected to Foxglove WebSocket")

    # Prepare the subscription message (changed op to 'subscribe')
    subscribe_message = {
        "op": "subscribe",  # Using 'subscribe' instead of 'advertise'
        "channelIds": [66, 65, 64, 63, 62, 69, 70, 71]  # List of channel IDs you want to subscribe to
    }

    # Send the subscription message to the server
    ws.send(json.dumps(subscribe_message))
    print("Subscription message sent: ", subscribe_message)


def connect_websocket():
    ws = websocket.WebSocketApp(
        foxglove_url,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
        on_open=on_open,
        header=["Sec-WebSocket-Protocol: foxglove.websocket.v1"]
    )
    ws.run_forever()


# Start WebSocket connection in a background thread
if st.button("Connect to Foxglove"):
    thread = threading.Thread(target=connect_websocket, daemon=True)
    # Add the script run context to the thread
    add_script_run_ctx(thread)
    thread.start()

# Show the direct Foxglove Studio link
foxglove_link = "https://app.foxglove.dev/canonicar/view?ds=foxglove-websocket&ds.url=ws://5.29.228.0:8765"
st.markdown(f"[Open in Foxglove Studio]({foxglove_link})")

# Display messages by topic
st.header("Messages by Topic")

# Filter topics based on user input
visible_topics = [
    topic for topic in st.session_state.messages_by_topic.keys()
    if not topic_filter or topic_filter.lower() in topic.lower()
]

if not visible_topics:
    st.info("No messages received yet or no topics match your filter.")
else:
    # Display messages for each topic in an expandable section
    for topic in sorted(visible_topics):
        with st.expander(f"Topic: {topic}"):
            for i, msg in enumerate(st.session_state.messages_by_topic[topic]):
                st.json(msg)
                if i < len(st.session_state.messages_by_topic[topic]) - 1:
                    st.divider()
