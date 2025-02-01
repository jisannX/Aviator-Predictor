from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load pre-trained model
model = pickle.load(open("model/aviator_predictor.pkl", "rb"))

@app.route('/predict', methods=['POST'])
def predict():
    try:
            data = request.get_json()
                    features = np.array(data["features"]).reshape(1, -1)
                            prediction = model.predict(features)
                                    return jsonify({"prediction": prediction.tolist()})
                                        except Exception as e:
                                                return jsonify({"error": str(e)})

                                                if __name__ == '__main__':
                                                    app.run(host="0.0.0.0", port=5000)