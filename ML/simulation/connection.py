import os
import sys
import carla
from simulation.settings import PORT, TIMEOUT, HOST


class ClientConnection:
    def __init__(self, town):
        self.client = None
        self.town = town

    def setup(self):
        try:
            # Connecting to the  Server
            print("Trying to connect to the server on port: {}".format(PORT))
            self.client = carla.Client(HOST, PORT)
            self.client.set_timeout(TIMEOUT)
            print(
                "Connected to the server successfully!\n Port: {}\n Town: {}".format(
                    PORT, self.town
                )
            )
            self.world = self.client.load_world(self.town)
            self.world.set_weather(carla.WeatherParameters.CloudyNoon)
            return self.client, self.world

        except Exception as e:
            print("Failed to make a connection with the server: {}".format(e))
            self.error()

    # An error method: prints out the details if the client failed to make a connection
    def error(self):
        print("\nClient version: {}".format(self.client.get_client_version()))
        print("Server version: {}\n".format(self.client.get_server_version()))

        if self.client.get_client_version != self.client.get_server_version:
            print(
                "There is a Client and Server version mismatch! Please install or download the right versions."
            )


if __name__ == "__main__":
    connection = ClientConnection("Town10HD")
    connection.setup()
