import os # this inports the os info of the local
import torch
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler

from src.models.lstm_model import DynamicPricingLSTM

def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


app = Flask(__name__)
CORS(app)

# Global variables to hold data and model
MODEL = None
SCALER = None
LATEST_SEQUENCE = None
INPUT_SIZE = None

def init_app():
    global MODEL, SCALER, LATEST_SEQUENCE, INPUT_SIZE
    
    # Load the data to fit the scaler and get the latest sequence
    df = pd.read_csv('data/synthetic_data.csv')
    feature_cols = ['m_t', 'c_t', 'S_t'] + [c for c in df.columns if 'p_t_seg' in c or 'd_t_seg' in c]
    data_values = df[feature_cols].values
    
    SCALER = StandardScaler()
    scaled_data = SCALER.fit_transform(data_values)
    
    seq_length = 7
    X_seq, _ = create_sequences(scaled_data, seq_length)
    
    # Grab the very last sequence to use as a base for live predictions
    LATEST_SEQUENCE = X_seq[-1] # shape (7, num_features)
    INPUT_SIZE = LATEST_SEQUENCE.shape[1]
    
    # Load Model
    MODEL = DynamicPricingLSTM(input_size=INPUT_SIZE, hidden_size=32, num_layers=2, num_segments=3)
    if os.path.exists('models_saved/lstm_model.pth'):
        MODEL.load_state_dict(torch.load('models_saved/lstm_model.pth'))
    MODEL.eval()

# Initialize immediately for Flask app context
init_app()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/notebooks/<path:filename>')
def serve_notebooks(filename):
    return send_from_directory('notebooks', filename)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        m_t = float(data.get('m_t', 0.5))
        c_t = float(data.get('c_t', 100))
        
        # We take the latest sequence and modify the last time step with user input
        # to simulate the "current" conditions
        modified_sequence = LATEST_SEQUENCE.copy()
        
        # To modify it properly, we need to inverse transform, modify, and re-transform
        # Or just modify it in raw space. Let's do raw space modification for the last step.
        raw_last_step = SCALER.inverse_transform(modified_sequence[-1].reshape(1, -1))[0]
        
        # Indices: m_t is 0, c_t is 1
        raw_last_step[0] = m_t
        raw_last_step[1] = c_t
        
        # Re-scale
        scaled_last_step = SCALER.transform(raw_last_step.reshape(1, -1))[0]
        modified_sequence[-1] = scaled_last_step
        
        # Prepare for model
        x_tensor = torch.tensor(modified_sequence, dtype=torch.float32).unsqueeze(0) # (1, 7, features)
        
        with torch.no_grad():
            prices = MODEL(x_tensor).numpy()[0]
            
        return jsonify({
            'status': 'success',
            'prices': {
                'segment_1': round(float(prices[0]), 2),
                'segment_2': round(float(prices[1]), 2),
                'segment_3': round(float(prices[2]), 2),
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
