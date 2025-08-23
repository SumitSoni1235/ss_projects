from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

model = pickle.load(open("LinearRegression.pkl", 'rb'))

datas = pd.read_csv("c:\\Users\\sumit\\Documents\\cleaned_car_data\\cleaned_data.csv")
print(datas)
@app.route('/')
def index():
    name_list = sorted(datas["name"].unique())
    year_list = sorted(datas["year"].unique(), reverse=True)
    km_driven_list = sorted(datas["km_driven"].unique())
    fuel_list = datas["fuel"].unique()
    seller_type_list = sorted(datas["seller_type"].unique())
    transmission_list = sorted(datas["transmission"].unique())
    owner_list = sorted(datas["owner"].unique())

    return render_template("app.html",
                           name=name_list,
                           year=year_list,
                           km_driven=km_driven_list,
                           fuel=fuel_list,
                           seller_type=seller_type_list,
                           transmission=transmission_list,
                           owner=owner_list)

@app.route('/predict', methods=['POST'])
def predict():
    name = request.form.get('name')
    year = int(request.form.get('year'))
    km_driven = int(request.form.get('km_driven'))
    fuel = request.form.get('fuel')
    seller_type = request.form.get('seller_type')
    transmission = request.form.get('transmission')
    owner = request.form.get('owner')

    input_df = pd.DataFrame([[name, year, km_driven, fuel, seller_type, transmission, owner]],
                            columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner'])
    prediction = model.predict(input_df)[0]
    print(prediction)
    return str(int(prediction))  # Return only the value for AJAX

if __name__ == "__main__":
    app.run(debug=True)