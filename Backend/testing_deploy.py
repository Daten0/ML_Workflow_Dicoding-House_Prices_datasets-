from flask import Flask, request, jsonify
import joblib
 
# Inisialisasi aplikasi Flask
app = Flask(__name__)
 
# Memuat model yang telah disimpan
joblib_model = joblib.load('../gbr_model.joblib') # Pastikan path file sesuai dengan penyimpanan Anda
 
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not request.json:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        if 'data' not in request.json:
            return jsonify({'error': 'No data field in JSON'}), 400
        
        data = request.json['data']
        
        # Validate and fix feature dimensions
        if isinstance(data, list) and len(data) > 0:
            # Check first sample to determine feature count
            sample_features = len(data[0]) if isinstance(data[0], list) else len(data)
            expected_features = 73  # Your model expects 73 features
            
            print(f"Received data with {sample_features} features")
            print(f"Model expects {expected_features} features")
            
            # Handle feature mismatch
            if sample_features == 76 and expected_features == 73:
                print("Adjusting feature dimensions: removing last 3 features")
                # Remove the last 3 features to match model expectations
                if isinstance(data[0], list):
                    adjusted_data = [sample[:73] for sample in data]
                else:
                    adjusted_data = [data[:73]]
                data = adjusted_data
            elif sample_features != expected_features:
                return jsonify({
                    'error': f'Feature dimension mismatch. Got {sample_features} features, but model expects {expected_features} features.'
                }), 400
        
        prediction = joblib_model.predict(data)
        return jsonify({'prediction': prediction.tolist()})
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
 
if __name__ == '__main__':
    app.run(debug=True)