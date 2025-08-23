import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
import pickle

car_dataset= pd.read_csv("C:\\Users\\sumit\\Documents\\CAR DETAILS FROM CAR DEKHO.csv")

car_dataset['name']=car_dataset['name'].str.split(' ').str.slice(0,3).str.join(' ')
car_dataset.replace({'owner':{"First Owner":'1st',"Second Owner":'2nd',"Third Owner":'3rd',"Fourth & Above Owner":'4th'}},inplace=True)
print(car_dataset)
car_dataset.to_csv("C:\\Users\\sumit\\Documents\\cleaned_car_data\\cleaned_data.csv")


X= car_dataset.drop(['selling_price'],axis=1)
Y=car_dataset['selling_price']

X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size=0.2, random_state=665)

ohe= OneHotEncoder()
ohe.fit(X[['name','fuel','seller_type','transmission','owner']])

column_trans= make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name','fuel','seller_type','transmission','owner']),remainder='passthrough')
L_model= LinearRegression()

pipe= make_pipeline(column_trans,L_model)

pipe.fit(X_train,Y_train)

pred= pipe.predict(X_test)
print(metrics.r2_score(Y_test,pred))


pickle.dump(pipe, open('LinearRegression.pkl', 'wb'))

prediction= pipe.predict(pd.DataFrame([['Maruti 800 AC',2007,200000,'Petrol','Individual','Manual','Second Owner']],
  columns=['name','year','km_driven','fuel','seller_type','transmission','owner']))

print(prediction)


# L_model.fit(X_train,Y_train)

# predict= L_model.predict(X_train)

# accuracy= metrics.r2_score(Y_train,predict)

# print("accuracy_score", accuracy)

# plt.scatter(Y_train,predict)
# plt.xlabel("actual")
# plt.ylabel("predicted")
# # plt.show()

# print(X_train.head())
# input_data= pd.DataFrame([[2017,700000,0,0,0,0]],columns=["year","km_driven","fuel","seller_type","transmission","owner"])


# print(L_model.predict(input_data))
