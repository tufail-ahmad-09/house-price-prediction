from flask import Flask,render_template,request
import pandas as pd
import pickle

app = Flask(__name__)

with open('house_price_model.pkl', 'rb')as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    try:
        medinc = request.form['medinc'].strip()
        houseage = request.form['houseage'].strip()

        
        medinc = ''.join(filter(str.isdigit, medinc))
        houseage = ''.join(filter(str.isdigit, houseage))

        medinc = float(medinc) if medinc else 0
        houseage = float(houseage) if houseage else 0

    
        


        # Prepare the data for prediction
        new_data = pd.DataFrame({
            'MedInc': [medinc],  
            'HouseAge': [houseage] 
        })

        # Make the prediction
        predictions = model.predict(new_data)[0]

        # Render the prediction result on a new page
        return render_template('result.html', predictions=predictions)

    except ValueError as ve:
        return f"Value error: {ve}"
    except Exception as e:
        return f"An error occurred: {e}"

    
    

if __name__ == "__main__":
    app.run(debug=True)
