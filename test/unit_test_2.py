import unittest
import os
import re
import json
import subprocess
import time
import requests
import warnings
from unittest.mock import patch
import securebank.app
from securebank.app import app as application


# Suppress DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class TestApp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Setup test environment: Start Docker container and determine exposed port."""
        cls.container_name = "test_container"
        cls.image_name = "test_image:latest"
        cls.port = 5000  # Default port if EXPOSE is not found
        cls.test_data = {
            "trans_date_trans_time": "2024-09-15 12:34:56",
            "cc_num": 38859492057661,
            "unix_time": 1694775296,
            "merchant": "fraud_Lind-Buckridge",
            "category": "entertainment",
            "amt": 50.75,
            "merch_lat": 43.150704,
            "merch_long": -112.154481
        }

        # Build the Docker image
        #build_result = subprocess.run(["docker", "build", "-t", cls.image_name, "."], capture_output=True, text=True)
        build_result = subprocess.run(["docker", "build", "-t", cls.image_name, "securebank"],
    capture_output=True, text=True
)
        assert build_result.returncode == 0, f"Docker build failed: {build_result.stderr}"

        # Extract exposed port from Dockerfile if exists
        if os.path.isfile("Dockerfile"):
            with open("Dockerfile", "r") as file:
                content = file.read()
            ports = re.findall(r"EXPOSE\s+(\d+)", content)
            if ports:
                cls.port = ports[0]

        # Start the container
        run_result = subprocess.run([
            "docker", "run", "-d", "-p", f"{cls.port}:{cls.port}",
            "--name", cls.container_name, cls.image_name
        ], capture_output=True, text=True)

        assert run_result.returncode == 0, f"Docker failed to start: {run_result.stderr}"

        # Wait for the container to be ready
        time.sleep(3)

        # Create a Flask test client
        cls.app = application.test_client()
        cls.app.testing = True

    def test_check_methods_exist_2(self):
        """Ensure the 'predict' method exists in the app."""

        routes = [rule.rule for rule in application.url_map.iter_rules()]
        self.assertIn("/predict", routes, "No /predict route found in Flask app (-40 pts)")

    def test_predict_call_success(self):
        """Test POST request to the /predict endpoint."""
        # Send a POST request to the /predict endpoint
        response = self.app.post('/predict', json=self.test_data)

        # Assert the response status code
        self.assertEqual(response.status_code, 200, f"Unexpected status code: {response.status_code}")

        # Ensure response contains valid JSON
        try:
            response_json = response.get_json()
        except Exception as e:
            self.fail(f"Response is not valid JSON: {response.data}. Error: {str(e)}")
        finally:
            response_json = response.get_json()

        print("This is the response:", response_json)

        # Ensure response is either a dictionary or an integer
        self.assertTrue(isinstance(response_json, (dict, int, str)), f"Response type is invalid: {type(response_json)}")

        # If response is a dictionary, check for a valid prediction value
        if isinstance(response_json, dict):
            self.assertIn("prediction", response_json, "Response JSON does not contain 'prediction' key")
            self.assertTrue(response_json["prediction"] in [0, 1, True, False, "True", "False", "Fraudulent", "Fraud", "Legitimate"],
                            f"Invalid prediction value: {response_json['prediction']}")

        # If response is an integer, ensure it is either 0 or 1
        elif isinstance(response_json, int):
            self.assertIn(response_json, [0, 1], f"Invalid integer response: {response_json}")

        # If response is an integer, ensure it is either 0 or 1
        elif isinstance(response_json, str):
            self.assertIn(response_json, ["True", "False"], f"Invalid integer response: {response_json}")


    def test_readme_exist(self):
        """Ensure README.md exists and display its contents."""
        readme_path = os.path.join("securebank", "README.md")
        self.assertTrue(os.path.isfile(readme_path) or os.path.isfile("README.md"), f"Missing README.md at {readme_path} (-20 pts)")
        if not os.path.isfile(readme_path):
            readme_path = "README.md"
        with open(readme_path, "r") as file:
            content = file.read()
        self.assertTrue('curl -X POST -H "Content-Type: application/json"' in content)
        self.assertTrue('docker build -t' in content)
        self.assertTrue('docker run -p 5000:5000' in content)
        
    def test_dockerfile_exist(self):
        """Ensure Dockerfile exists in securebank/ directory."""
        dockerfile_path = os.path.join("securebank", "Dockerfile")
        self.assertTrue(os.path.isfile(dockerfile_path), f"Missing Dockerfile at {dockerfile_path} (-40 pts)")



    def test_docker_container_build(self):
        """Verify Docker image builds successfully from securebank/."""
        result = subprocess.run(
            ["docker", "build", "-t", "test_image", "securebank"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print("Build stderr:\n", result.stderr)
            print("Build stdout:\n", result.stdout)
        self.assertEqual(result.returncode, 0, "Docker build failed")

    @patch("requests.get")
    def test_mocked_docker_ping(self, mock_get):
        """Mock test for Docker ping endpoint."""
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = "OK"
        response = requests.get(f"http://127.0.0.1:{self.port}/")
        time.sleep(1)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text, "OK")

    def test_docker_endpoint(self):
        """Test the running Docker container's API endpoint."""
        url = f"http://127.0.0.1:{self.port}/predict"
        try:
            response = requests.post(url, json=self.test_data, headers={"Content-Type": "application/json"})
            self.assertEqual(response.status_code, 200, f"Unexpected response: {response.text}")
            print(f"Endpoint output: {response.text}")
        except requests.exceptions.ConnectionError as e:
            self.fail(f"Connection failed: {e}")

    @classmethod
    def tearDownClass(cls):
        """Cleanup: Stop and remove the Docker container."""
        subprocess.run(["docker", "stop", cls.container_name], capture_output=True, text=True)
        subprocess.run(["docker", "rm", cls.container_name], capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main(buffer=True)
