from flask import Flask, render_template, request
import numpy as np
import pickle

# Load the trained Machine Learning model
with open("MLR_Model", "rb") as f:
    model = pickle.load(f)

# Create Flask application
app = Flask(__name__)


# Home page
@app.route("/")
def home():
    return render_template(
        "index.html",
        prediction_text=None,
        error_message=None
    )


# Predict startup profit
@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Get input values from index.html
        rd_spend = float(request.form["rd_spend"])
        administration = float(request.form["administration"])
        marketing_spend = float(request.form["marketing_spend"])
        state = int(request.form["state"])

        # Validate investment values
        investments = np.array([
            rd_spend,
            administration,
            marketing_spend
        ])

        if not np.isfinite(investments).all():
            raise ValueError(
                "Please enter valid investment amounts."
            )

        if np.any(investments < 0):
            raise ValueError(
                "Investment amounts cannot be negative."
            )

        # Validate state
        if state not in [0, 1, 2]:
            raise ValueError(
                "Please select a valid state."
            )

        # Prepare model input
        input_data = np.array(
            [[
                rd_spend,
                administration,
                marketing_spend,
                state
            ]],
            dtype=float
        )

        print("Input data:", input_data)
        print("Input shape:", input_data.shape)

        # Predict profit
        prediction = model.predict(input_data)

        # Convert the model output into a NumPy array
        prediction_array = np.asarray(prediction)

        print("Raw prediction:", prediction_array)
        print("Prediction shape:", prediction_array.shape)

        # Ensure there is exactly one prediction
        if prediction_array.size != 1:
            raise ValueError(
                "The model did not return a single profit value. "
                "Check your model training code."
            )

        # Extract the predicted profit
        prediction_text = float(prediction_array.item())

        if not np.isfinite(prediction_text):
            raise ValueError(
                "The model returned an invalid profit value."
            )

        print("Predicted profit:", prediction_text)

        # Return the prediction to index.html
        return render_template(
            "index.html",
            prediction_text=prediction_text,
            error_message=None
        )

    except (ValueError, KeyError, TypeError) as error:

        print("Prediction error:", error)

        return render_template(
            "index.html",
            prediction_text=None,
            error_message=str(error)
        )

    except Exception as error:

        app.logger.exception("Prediction failed")

        return render_template(
            "index.html",
            prediction_text=None,
            error_message=(
                "Unable to predict profit. "
                f"Check the terminal for details: {error}"
            )
        )


# Run Flask application
if __name__ == "__main__":
    app.run(debug=True)