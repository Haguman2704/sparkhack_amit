from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__,static_folder="E:\model\ML-model-deployment-part-2-main\static")

# Load the trained model
model = joblib.load('model.pkl')

# Define routes
@app.route('/')
def index():
    return render_template('settings.html')
@app.route('/regform.html', methods=['POST','GET'])
def reg():
    # Get reg details from the form
    house_number = int(request.form['house_number'])
    date = pd.to_datetime(request.form['date'], dayfirst=True)
    day_of_week = date.dayofweek
    month = date.month
    season = (date.month % 12 + 3) // 3
    date_ordinal = date.toordinal()

    # Make prediction
    prediction = model.predict([[house_number, day_of_week, month, season, date_ordinal]])

    # Render result template
    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
