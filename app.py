from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # Get form data
        weight = float(request.form['weight'])
        height = float(request.form['height'])
        bmi = float(request.form['bmi'])
        gender = int(request.form['gender'])  # assume 0 for female, 1 for male
        age = int(request.form['age'])
        bmi_case = request.form['bmi_case']
        d={'mild thinness':0,'moderate thinness':1,'normal':2,'obese':3,'over weight':4,'severe thinness':5,'severe obese':6}
        # Convert to input format for the model
        bmi_case=d[bmi_case.lower()]
        features = np.array([[weight, height, bmi, gender, age, bmi_case,height/weight,bmi/age]])
        # Predict recommendation
        recommendation = model.predict(features)
        return render_template('index.html', recommendation=int(round(recommendation[0],0)))
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
