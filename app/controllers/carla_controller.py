from flask import Blueprint, Response, jsonify
from app.models.carla_factory import CarlaFactory
from app.models.carla_model import CarlaModel



carla_controller = Blueprint('carla_controller', __name__)

# Initialize the CARLA model
carla_model = CarlaModel()
carla_factory = CarlaFactory(carla_model)


@carla_controller.route('/start', methods=['POST'])
def start():
    try:
        carla_model.connect()          
        return jsonify({"status": "success", "message": "CARLA connected, vehicle spawned, and camera setup."}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# spawn car with a camera attached
@carla_controller.route('/spawn_car', methods=['POST'])
def spawn_car():
    try:
        vehicle_model = carla_factory.create_vehicle()
        vehicle = vehicle_model.spawn_vehicle() 
        camera_model = carla_factory.create_camera(vehicle)
        camera = camera_model.setup_camera()
        return jsonify({"status": "success", "message": "Vehicle spawned.", "vehicle":{vehicle}}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# For now show only the first camera
@carla_controller.route('/video_feed')
def video_feed():
    """Stream video to the page."""
    def generate():
        while True:
            frame = carla_model.cameras[0].get_frame()  # Get the latest frame from the camera
            if frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
