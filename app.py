from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# Function to predict the Post Office based on the address and Pin Code
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    address = data['address']
    pincode = data['pincode']

    # Encode the address using the trained label encoder
    address_encoded = label_encoder.transform([address])[0]
    
    # Predict the Post Office
    prediction = model.predict([[pincode, address_encoded]])
    post_office = label_encoder.inverse_transform(prediction)[0]

    return jsonify({'post_office': post_office})

if __name__ == '__main__':
    app.run(debug=True)
