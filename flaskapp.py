from flask import Flask, request, render_template_string, jsonify
import pandas as pd
import joblib
import json
import os
from datetime import datetime
import numpy as np


app = Flask(__name__)


# Load the pretrained pipeline (including preprocessing)
model = joblib.load('crop_yield_best_model2.pkl')


HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'crop_predictions_history.json')


def load_history():
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r') as f:
                content = f.read()
                if content.strip():  # Check if file is not empty
                    try:
                        return json.loads(content)
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error: {e}. Creating backup and starting fresh.")
                        # Backup corrupted file
                        backup_file = HISTORY_FILE + '.backup'
                        with open(backup_file, 'w') as backup:
                            backup.write(content)
                        print(f"Corrupted file backed up to: {backup_file}")
                        # Return empty list and file will be recreated on next save
                        return []
        return []
    except Exception as e:
        print(f"Error loading history: {e}")
        return []


def save_history(history):
    try:
        # Write to a temporary file first
        temp_file = HISTORY_FILE + '.tmp'
        with open(temp_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        # On Windows, need to delete the original file first before renaming
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
        
        os.rename(temp_file, HISTORY_FILE)
        
        print(f"Successfully saved {len(history)} records to history")
        print(f"History file location: {HISTORY_FILE}")
    except Exception as e:
        print(f"Error saving history: {e}")
        import traceback
        traceback.print_exc()
        # If atomic save fails, try direct write as fallback
        try:
            print("Attempting direct write as fallback...")
            with open(HISTORY_FILE, 'w') as f:
                json.dump(history, f, indent=2)
            print("Direct write successful")
        except Exception as e2:
            print(f"Direct write also failed: {e2}")


form_template = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AgriPredict - AI-Powered Crop Yield Forecasting</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
      background: linear-gradient(135deg, #fad0c4 0%, #ffd1ff 100%);
      min-height: 100vh;
      color: #3b223a;
      padding: 20px;
    }
    .header-bar {
      background: rgba(255, 200, 220, 0.39);
      backdrop-filter: blur(16px);
      border: 1px solid rgba(255, 170, 210, 0.17);
      border-radius: 16px;
      padding: 20px 30px;
      margin-bottom: 30px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      box-shadow: 0 4px 24px 0 rgba(220, 105, 180, 0.12);
    }
    .logo-section { display: flex; align-items: center; gap: 15px; }
    .logo-icon {
      width: 50px; height: 50px;
      background: linear-gradient(135deg, #fbbedc 0%, #fad0c4 100%);
      border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 24px;
      box-shadow: 0 4px 12px 0 rgba(255,182,213,0.19);
    }
    .logo-text h1 { font-size: 28px; font-weight: 700; color: #db2072; margin-bottom: 2px; }
    .logo-text p { font-size: 13px; color: #9e4784; font-weight: 400; }
    .nav-buttons { display: flex; gap: 10px; }
    .nav-btn {
      background: rgba(255, 190, 220, 0.39);
      border: 1px solid rgba(255, 95, 187, 0.11);
      color: #c84b7d;
      padding: 12px 24px; border-radius: 10px; cursor: pointer;
      font-size: 14px; font-weight: 500;
      transition: all 0.3s ease;
      display: flex; align-items: center; gap: 8px;
      box-shadow: 0 2px 12px rgba(255, 180, 230, 0.11);
    }
    .nav-btn:hover, .nav-btn.active {
      background: linear-gradient(135deg, #fbbedc 20%, #ffd6e0 100%);
      border-color: #f39fc3;
      color: #b51c67;
      box-shadow: 0 4px 14px rgba(255, 145, 207, 0.14);
    }
    .section-container { position: relative; min-height: 500px; }
    .main-container {
      max-width: 1400px; margin: 0 auto;
      display: grid; grid-template-columns: 1fr 1fr; gap: 24px;
      transition: opacity 0.4s, transform 0.4s;
    }
    .main-container.slide-in {
      opacity: 1;
      transform: translateX(0);
      pointer-events: auto;
      z-index: 2;
      position: relative;
    }
    .main-container.slide-out {
      opacity: 0;
      transform: translateX(-100%);
      pointer-events: none;
      z-index: 1;
      position: absolute;
      width: 100%;
      top: 0;
      left: 0;
    }
    .card {
      background: rgba(255, 235, 245, 0.39);
      backdrop-filter: blur(22px);
      border: 1px solid rgba(255, 124, 188, 0.11);
      border-radius: 20px;
      padding: 28px;
      box-shadow: 0 4px 24px rgba(255, 128, 187, 0.11);
    }
    .card-header { display: flex; align-items: center; gap: 15px; margin-bottom: 28px; }
    .card-icon {
      width: 48px; height: 48px;
      background: linear-gradient(135deg, #eaafc8 0%, #fde1ff 100%);
      border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 24px;
      box-shadow: 0 4px 10px rgba(255, 145, 207, 0.15);
    }
    .card-title h2 { font-size: 22px; font-weight: 600; color: #b51c67; margin-bottom: 4px; }
    .card-title p { font-size: 13px; color: #9e4784; }
    .form-group { margin-bottom: 20px; }
    .form-group label { display: block; margin-bottom: 8px; font-size: 14px; font-weight: 500; color: #bb3e7a; }
    .form-group input, .form-group select {
      width: 100%; padding: 12px 16px;
      background: rgba(255, 200, 220, 0.14);
      border: 1px solid rgba(255, 95, 187, 0.13); border-radius: 12px;
      color: #3d1150; font-size: 14px; transition: all 0.3s ease;
    }
    .form-group input:focus, .form-group select:focus {
      outline: none; border-color: #ea7ea0;
      box-shadow: 0 0 0 3px rgba(255, 190, 220, 0.08);
    }
    .form-group input::placeholder { color: #c07ea9; }
    .grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 20px; }
    .submit-btn {
      width: 100%;
      background: linear-gradient(135deg, #fcadc2 0%, #fea9de 100%);
      border: none; color: #fff;
      padding: 14px 24px; border-radius: 10px; cursor: pointer;
      font-size: 15px; font-weight: 600;
      display: flex; align-items: center; justify-content: center; gap: 10px;
      transition: all 0.3s ease;
      box-shadow: 0 4px 12px rgba(250, 167, 223, 0.09);
      margin-top: 24px;
    }
    .submit-btn:hover {
      background: linear-gradient(135deg, #ffcee4 0%, #eaafc8 100%);
      transform: translateY(-2px);
    }
    .result-box {
      background: linear-gradient(135deg, rgba(251, 190, 220, 0.15) 0%, rgba(255, 209, 255, 0.12) 100%);
      border: 1px solid rgba(255,190,220,0.16);
      border-radius: 16px;
      padding: 32px; text-align: center;
    }
    .result-label { font-size: 14px; color: #b989bb; margin-bottom: 12px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }
    .result-value { font-size: 56px; font-weight: 700; color: #fb5693; margin-bottom: 8px; text-shadow: 0 0 20px rgba(251, 86, 147, 0.11); }
    .result-unit { font-size: 18px; color: #c07ea9; font-weight: 500; }
    .quality-section {
      background: rgba(255, 190, 220, 0.21);
      border-radius: 14px; padding: 24px; margin-top: 24px;
      box-shadow: 0 2px 8px #f4d3ec2a; border: 1px solid #ffd6e03a;
    }
    .quality-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; }
    .quality-label { font-size: 13px; color: #c07ea9; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 500; }
    .quality-status { display: flex; align-items: center; gap: 12px; }
    .quality-text { font-size: 20px; font-weight: 600; color: #ea7ea0; }
    .quality-percentage { font-size: 28px; font-weight: 700; color: #db2072; }
    .progress-bar { width: 100%; height: 8px; background: rgba(255, 167, 223, 0.17); border-radius: 4px; overflow: hidden; position: relative; }
    .progress-fill { height: 100%; background: linear-gradient(90deg, #ea7ea0 0%, #f3a3bb 100%); border-radius: 4px; width: 0; transition: width 1.5s cubic-bezier(0.4, 0, 0.2, 1); }
    .recommendations { margin-top: 24px; }
    .rec-header { display: flex; align-items: center; gap: 10px; margin-bottom: 16px; font-size: 14px; color: #c874aa; font-weight: 500; }
    .rec-item { background: rgba(253, 136, 203, 0.07); border: 1px solid rgba(250, 167, 223, 0.11); border-radius: 10px; padding: 14px 16px; display: flex; align-items: center; gap: 12px; color: #b5457c; font-size: 14px; line-height: 1.5; }
    .rec-icon { color: #ea7ea0; font-size: 18px; }
    .empty-state { text-align: center; padding: 60px 20px; }
    .empty-state-icon { font-size: 64px; margin-bottom: 16px; opacity: 0.20; }
    .empty-state p { color: #c07ea9; font-size: 15px; }
    .error-box { background: rgba(255, 99, 132, 0.13); border: 1px solid rgba(234,126,160,0.22); border-radius: 14px; color: #e7588f; font-size: 14px; text-align: center; padding: 16px; }
    /* History styles: update inner table colors to match */
    #history-content table th, #history-content table td { color: #b5457c !important; }
    #history-content table th { background: rgba(237, 113, 183, 0.06) !important; }
    #history-content table tr { background: rgba(255,255,255,0.02) !important; }
    @media (max-width: 1024px) { .main-container { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <div class="header-bar">
    <div class="logo-section">
      <div class="logo-icon">🌸</div>
      <div class="logo-text">
        <h1>AgriPredict</h1>
        <p>AI-Powered Crop Yield Forecasting</p>
      </div>
    </div>
    <div class="nav-buttons">
      <button class="nav-btn active" onclick="showTab('predict')">📈 Predict Yield</button>
      <button class="nav-btn" onclick="showTab('history')">🕐 History</button>
    </div>
  </div>

  <div class="section-container">
    <div id="predict-section" class="main-container slide-in">
      <div class="card">
        <div class="card-header">
          <div class="card-icon">🌸</div>
          <div class="card-title">
            <h2>Soil Analysis</h2>
            <p>Enter your field's NPK values</p>
          </div>
        </div>
        <form method="POST">
          <div class="form-group">
            <label for="Soil_pH">Soil pH</label>
            <input id="Soil_pH" name="Soil_pH" type="number" step="any" placeholder="6.5" value="{{ request.form.get('Soil_pH', '') }}" required />
          </div>
          <div class="form-group">
            <label for="Temperature">Temperature (°C)</label>
            <input id="Temperature" name="Temperature" type="number" step="any" placeholder="15.2" value="{{ request.form.get('Temperature', '') }}" required />
          </div>
          <div class="form-group">
            <label for="Humidity">Humidity (%)</label>
            <input id="Humidity" name="Humidity" type="number" step="any" placeholder="80" value="{{ request.form.get('Humidity', '') }}" required />
          </div>
          <div class="form-group">
            <label for="Wind_Speed">Wind Speed (km/h)</label>
            <input id="Wind_Speed" name="Wind_Speed" type="number" step="any" placeholder="6.7" value="{{ request.form.get('Wind_Speed', '') }}" required />
          </div>
          <div class="grid-3">
            <div class="form-group">
              <label for="N">Nitrogen (N)</label>
              <input id="N" name="N" type="number" step="any" placeholder="63" value="{{ request.form.get('N', '') }}" required />
            </div>
            <div class="form-group">
              <label for="P">Phosphorus (P)</label>
              <input id="P" name="P" type="number" step="any" placeholder="60" value="{{ request.form.get('P', '') }}" required />
            </div>
            <div class="form-group">
              <label for="K">Potassium (K)</label>
              <input id="K" name="K" type="number" step="any" placeholder="55" value="{{ request.form.get('K', '') }}" required />
            </div>
          </div>
          <div class="form-group">
            <label for="Soil_Quality">Soil Quality</label>
            <input id="Soil_Quality" name="Soil_Quality" type="number" step="any" placeholder="59.3" value="{{ request.form.get('Soil_Quality', '') }}" required />
          </div>
          <div class="form-group">
            <label for="Crop_Type">Crop Type</label>
            <input id="Crop_Type" name="Crop_Type" type="text" placeholder="Soybean" value="{{ request.form.get('Crop_Type', '') }}" required autocomplete="off" />
          </div>
          <button type="submit" class="submit-btn">
            <span>📊</span>
            Predict Yield
          </button>
        </form>
      </div>
      <div class="card">
        <div class="card-header">
          <div class="card-icon">📊</div>
          <div class="card-title">
            <h2>Prediction Results</h2>
            <p>AI-powered yield forecast</p>
          </div>
        </div>
        {% if prediction is not none %}
        <div class="result-box">
          <div class="result-label">Predicted Yield</div>
          <div class="result-value" id="yield-value">0</div>
          <div class="result-unit">tons/ha</div>
        </div>
        <script>
          setTimeout(() => {
            const targetYield = {{ prediction }};
            const yieldElement = document.getElementById('yield-value');
            let currentYield = 0;
            const step = targetYield / 50;
            
            const animateYield = () => {
              if (currentYield < targetYield) {
                currentYield = Math.min(currentYield + step, targetYield);
                yieldElement.textContent = currentYield.toFixed(2);
                requestAnimationFrame(animateYield);
              }
            };
            
            animateYield();
          }, 300);
        </script>
        <div class="quality-section">
          <div class="quality-header">
            <span class="quality-label">Soil Quality Assessment</span>
            <div class="quality-status">
              <span class="quality-text">{% if soil_quality >= 70 %}Excellent{% elif soil_quality >= 50 %}Good{% else %}Fair{% endif %}</span>
              <span class="quality-percentage" id="quality-percentage">0%</span>
            </div>
          </div>
          <div class="progress-bar">
            <div class="progress-fill" id="progress-fill"></div>
          </div>
          <script>
            setTimeout(() => {
              const targetPercentage = {{ soil_quality }};
              const progressFill = document.getElementById('progress-fill');
              const percentageText = document.getElementById('quality-percentage');
              let currentPercentage = 0;
              const animate = () => {
                if (currentPercentage < targetPercentage) {
                  currentPercentage += 1;
                  percentageText.textContent = `${currentPercentage}%`;
                  progressFill.style.width = `${currentPercentage}%`;
                  requestAnimationFrame(animate);
                }
              };
              animate();
            }, 300);
          </script>
        </div>
        <div class="recommendations">
          <div class="rec-header">
            <span>💡</span>
            <span>Recommendations</span>
          </div>
          <div class="rec-item">
            <span class="rec-icon">✓</span>
            <span>{% if soil_quality >= 70 %}Optimal conditions detected. Continue current soil management practices.{% elif soil_quality >= 50 %}Good soil conditions. Consider minor nutrient adjustments for optimal yield.{% else %}Soil quality needs improvement. Consider soil amendments and nutrient supplementation.{% endif %}</span>
          </div>
        </div>
        {% elif error %}
        <div class="error-box">{{ error }}</div>
        {% else %}
        <div class="empty-state">
          <div class="empty-state-icon">📈</div>
          <p>Enter soil and crop data to get yield prediction</p>
        </div>
        {% endif %}
      </div>
    </div>
    <div id="history-section" class="main-container slide-out">
      <div class="card" style="grid-column: 1 / -1;">
        <div class="card-header">
          <div class="card-icon">🕐</div>
          <div class="card-title">
            <h2>Prediction History</h2>
            <p>Your past predictions</p>
          </div>
        </div>
        <div id="history-content">
          <div class="empty-state">
            <div class="empty-state-icon">📋</div>
            <p>Loading history...</p>
          </div>
        </div>
      </div>
    </div>
  </div>
  <script>
    function showTab(tab) {
      const predictSection = document.getElementById('predict-section');
      const historySection = document.getElementById('history-section');
      const buttons = document.querySelectorAll('.nav-btn');
      if (tab === 'predict') {
        historySection.classList.add('slide-out');
        historySection.classList.remove('slide-in');
        predictSection.classList.remove('slide-out');
        predictSection.classList.add('slide-in');
        buttons[0].classList.add('active');
        buttons[1].classList.remove('active');
      } else {
        predictSection.classList.add('slide-out');
        predictSection.classList.remove('slide-in');
        historySection.classList.remove('slide-out');
        historySection.classList.add('slide-in');
        buttons[0].classList.remove('active');
        buttons[1].classList.add('active');
        loadHistory();
      }
    }
    async function loadHistory() {
      try {
        const response = await fetch('/api/history');
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        
        // Check if there's an error in the response
        if (data.error) {
          throw new Error(data.error);
        }
        
        const history = data;
        const content = document.getElementById('history-content');
        
        console.log('History loaded:', history); // Debug output
        
        if (!history || history.length === 0) {
          content.innerHTML = `
            <div class="empty-state">
              <div class="empty-state-icon">📋</div>
              <p>No predictions yet</p>
            </div>
          `;
          return;
        }
        let html = '<div style="overflow-x: auto;"><table style="width: 100%; border-collapse: collapse;">';
        html += `
          <thead>
            <tr style="border-bottom: 1px solid rgba(255,124,188,0.14);">
              <th style="padding: 12px; text-align: left;">Date</th>
              <th style="padding: 12px; text-align: left;">Crop</th>
              <th style="padding: 12px; text-align: left;">N-P-K</th>
              <th style="padding: 12px; text-align: left;">Soil pH</th>
              <th style="padding: 12px; text-align: left;">Soil Quality</th>
              <th style="padding: 12px; text-align: left;">Yield</th>
            </tr>
          </thead>
          <tbody>
        `;
        history.forEach((record) => {
          html += `
            <tr style="border-bottom: 1px solid rgba(255,124,188,0.12);">
              <td style="padding: 12px;">${record.date || 'N/A'}</td>
              <td style="padding: 12px;">${record.Crop_Type || 'N/A'}</td>
              <td style="padding: 12px;">${record.N || 0}-${record.P || 0}-${record.K || 0}</td>
              <td style="padding: 12px;">${record.Soil_pH || 'N/A'}</td>
              <td style="padding: 12px;">${record.Soil_Quality || 'N/A'}</td>
              <td style="padding: 12px; color: #fb5693; font-weight: 600;">${record.yield || 'N/A'} tons/ha</td>
            </tr>
          `;
        });
        html += '</tbody></table></div>';
        content.innerHTML = html;
      } catch (error) {
        console.error('Error loading history:', error);
        const content = document.getElementById('history-content');
        content.innerHTML = `
          <div class="empty-state">
            <div class="empty-state-icon">⚠️</div>
            <p>Error loading history: ${error.message}</p>
            <p style="font-size: 12px; margin-top: 8px;">Check browser console for details</p>
          </div>
        `;
      }
    }
  </script>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    soil_quality = None
    error = None
    if request.method == 'POST':
        try:
            input_dict = {
                'Soil_pH': float(request.form['Soil_pH']),
                'Temperature': float(request.form['Temperature']),
                'Humidity': float(request.form['Humidity']),
                'Wind_Speed': float(request.form['Wind_Speed']),
                'N': float(request.form['N']),
                'P': float(request.form['P']),
                'K': float(request.form['K']),
                'Soil_Quality': float(request.form['Soil_Quality']),
                'Crop_Type': request.form['Crop_Type'].strip()
            }
            
            print(f"Input data: {input_dict}")
            
            input_df = pd.DataFrame([input_dict])
            print("Making prediction...")
            prediction_value = model.predict(input_df)[0]
            print(f"Raw prediction value: {prediction_value}, type: {type(prediction_value)}")
            
            # Ensure prediction is a valid number
            if prediction_value is None or not isinstance(prediction_value, (int, float, np.number)):
                raise ValueError(f"Invalid prediction value: {prediction_value}")
            
            prediction = round(float(prediction_value), 2)
            soil_quality = round(input_dict['Soil_Quality'])
            
            print(f"Rounded prediction: {prediction}")
            
            # Save to history - ensure all values are JSON serializable
            history = load_history()
            record = {
                'Soil_pH': float(input_dict['Soil_pH']),
                'Temperature': float(input_dict['Temperature']),
                'Humidity': float(input_dict['Humidity']),
                'Wind_Speed': float(input_dict['Wind_Speed']),
                'N': float(input_dict['N']),
                'P': float(input_dict['P']),
                'K': float(input_dict['K']),
                'Soil_Quality': float(input_dict['Soil_Quality']),
                'Crop_Type': str(input_dict['Crop_Type']),
                'yield': float(prediction),
                'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            print(f"Record to save: {record}")
            print(f"Record yield value: {record['yield']}, type: {type(record['yield'])}")
            
            history.insert(0, record)
            
            print(f"About to save history with {len(history)} records")
            save_history(history)
            print("History save completed")
            
        except Exception as e:
            error = f"Invalid input or error during prediction: {e}"
            print(f"Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            
    return render_template_string(form_template, prediction=prediction, soil_quality=soil_quality, error=error)

@app.route('/api/history')
def get_history():
    try:
        history = load_history()
        print(f"History file path: {HISTORY_FILE}")
        print(f"File exists: {os.path.exists(HISTORY_FILE)}")
        print(f"Loaded {len(history)} history records")
        if history:
            print(f"First record: {history[0]}")
        return jsonify(history)
    except Exception as e:
        print(f"Error in get_history: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/debug')
def debug_history():
    """Debug route to check history file"""
    try:
        file_path = HISTORY_FILE
        exists = os.path.exists(file_path)
        
        debug_info = {
            "file_path": file_path,
            "file_exists": exists,
            "current_dir": os.getcwd()
        }
        
        if exists:
            with open(file_path, 'r') as f:
                content = f.read()
                debug_info["file_size"] = len(content)
                debug_info["file_content_preview"] = content[:500] if content else "EMPTY"
                try:
                    data = json.loads(content) if content.strip() else []
                    debug_info["records_count"] = len(data)
                    debug_info["data"] = data
                except json.JSONDecodeError as e:
                    debug_info["json_error"] = str(e)
        
        return jsonify(debug_info)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)