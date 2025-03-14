import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Load the pre-trained model
model = joblib.load('fish_market_model.pkl')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return redirect(url_for('run_model', filename=filename))
    return redirect(request.url)

@app.route('/run_model/<filename>')
def run_model(filename):
    try:
        # Read uploaded CSV
        data = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        # Assuming the CSV has features (X) and target (y)
        X = data.drop('Weight', axis=1)  # Replace 'target' with your actual target column name
        y = data['Weight']
        
        # Generate predictions
        predictions = model.predict(X)
        
        # Calculate metrics
        mae = mean_absolute_error(y, predictions)
        mse = mean_squared_error(y, predictions)
        r2 = r2_score(y, predictions)
        
        # Pass results to template
        return render_template('results.html', 
                             mae=round(mae, 2), 
                             mse=round(mse, 2), 
                             r2=round(r2, 2))
    except Exception as e:
        return f"Error processing file: {str(e)}"

if __name__ == '__main__':
    # Create uploads directory if not exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
    
