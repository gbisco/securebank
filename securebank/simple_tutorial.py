from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def hello_world():
    """
    Index route that returns a welcome message.
    Returns:
    Response with a welcome message.
    """
    return "Hello World!"

# Modify for Assignment 2 and Final Case Study
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Get JSON data from the request
    input_data = request.get_json()

    # Instantiate modules.model.Model object

    # Call modules.model.Model.predict(input_data)

    return "No prediction" 


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)