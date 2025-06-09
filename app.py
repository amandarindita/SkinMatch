from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)


nb_model = joblib.load('nb_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')


df = pd.read_csv('skincare_products.csv')

@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    if request.method == "POST":
        skin_type = request.form["skin_type"]
        problems = request.form["problems"]
        type_product = request.form["type_product"]

     
        input_text = f'{skin_type} {problems}'
        input_vec = vectorizer.transform([input_text])

       
        predicted_type = nb_model.predict(input_vec)[0]
        
      
        print(f'Input: {input_text}')
        print(f'Predicted Type: {predicted_type}')

        
        df_filtered = df[(df['Jenis Kulit'].str.lower() == skin_type.lower()) & 
                         (df['Permasalahan Kulit'].str.lower() == problems.lower()) & 
                         (df['Type'].str.lower() == type_product.lower())]

        recommendations = df_filtered.head(10).to_dict(orient='records')
            
    return render_template("index.html", recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
