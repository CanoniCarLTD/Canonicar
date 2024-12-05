# server.py
from flask import Flask, Response, render_template
import cv2
import threading
from carla_client import CarlaClient
import numpy as np

app = Flask(__name__)

# Global variables for video streaming
frame = None
lock = threading.Lock()

# Initialize CARLA client
carla_client = CarlaClient("5.29.228.0", 2000)  # Use local IP here
carla_client.connect()
carla_client.setup_camera()

# Function to update the frame


def carla_frame_callback(image):
    global frame
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))  # BGRA
    frame_bgr = array[:, :, :3]  # Convert to BGR
    with lock:
        _, buffer = cv2.imencode('.jpg', frame_bgr)
        frame = buffer.tobytes()


# Start CARLA streaming
carla_client.start_stream(carla_frame_callback)


@app.route('/')
def index():
    """Render the homepage."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Stream video to the page."""
    def generate():
        global frame
        while True:
            with lock:
                if frame is not None:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        carla_client.stop_camera()
