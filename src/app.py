from flask import Flask, request, jsonify
from predict import predict_sentiment

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    review = data.get("review", "")
    if not review:
        return jsonify({"error": "Review tidak boleh kosong"}), 400
    
    sentiment = predict_sentiment(review)
    return jsonify({"review": review, "sentiment": sentiment})

if __name__ == "__main__":
    app.run(debug=True)
