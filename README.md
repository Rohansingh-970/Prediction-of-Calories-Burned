import numpy as np
import pandas as pd
df = pd.read_csv('/content/exercisecombined.csv')
from google.colab import drive
drive.mount('/content/drive')
df.head()
df.shape
df['Gender'].value_counts()
a = df['Age'].min()
b = df['Age'].max()
print(a)
print(b)
print(df['Duration'].min(),df['Duration'].max())
print(df['Body_Temp'].min(),df['Body_Temp'].max())
a=df['Height'].min()
b = df['Height'].max()
print(a,b)
print(df['Heart_Rate'].min(),df['Heart_Rate'].max())
print(df['Weight'].min(),df['Weight'].max())
df.describe()
df.describe()
df['Gender'] = df['Gender'].map({'male':1,"female":0})
df['Gender']
df.head()
X = df.drop(['User_ID','Calories'],axis=1)
y =df['Calories'] #target variable
X.shape
y.shape
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)
#randomforest ---- bootstrapping,bagging
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
rf = RandomForestClassifier(n_estimators = 126, max_depth = 20, random_state = 43, max_leaf_nodes = 6, min_samples_split = 3, min_samples_leaf = 4)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
score = accuracy_score(y_test, y_pred)
print(score)
print(type(y_test))
X_train.shape
X_test.shape
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
LR = LinearRegression(fit_intercept=False,copy_X=True,n_jobs=10,positive=True)
LR.fit(X_train,y_train)
y_pred = LR.predict(X_test)
round(r2_score(y_test,y_pred)*100,2)
mean_squared_error(y_test,y_pred)
DCT = DecisionTreeRegressor(criterion="absolute_error",splitter="best",max_depth=3,min_samples_split=3,min_samples_leaf=2,random_state=42,max_leaf_nodes=3)
DCT.fit(X_train,y_train)
y_pred_2 = DCT.predict(X_test)
round(r2_score(y_test,y_pred_2)*100,2)
from xgboost import XGBRegressor
XGB = XGBRegressor(max_depth=20,max_leaves=10,n_jobs=50,random_state=52,min_child_weight=8,n_estimators=200,max_bin=2)
XGB.fit(X_train,y_train)
y_pred_3 = XGB.predict(X_test)
round(r2_score(y_test,y_pred_3)*100,2)
import pickle

pickle.dump(XGB, open('/content/XGB.pkl', 'wb'))
X_train.to_csv('X_train.csv')
pip install gradio
import pickle
import gradio as gr

# Load the trained model
model = pickle.load(open('/content/XGB.pkl', 'rb'))

def predict_calories(exercise_time, body_temp, Heart_rate, gender, age, height, weight):


    predicted_calories = model.predict([[exercise_time, body_temp, Heart_rate, gender, age, height, weight]])

    return predicted_calories[0]

# Create a Gradio interface
iface = gr.Interface(

    fn=predict_calories,
    theme = gr.themes.Monochrome(),
    inputs=[
        gr.components.Slider(1,30,label="Exercise Duration (minutes 1-30)",step=1),
        gr.components.Slider(36,42,label="Body Temperature (°C 36-42)"),
        gr.components.Slider(65,130,label="Heart Rate (BPM 65-130)",step=1),
        gr.components.Slider(0,1,label="Gender (1-Female,0-Male)",step=1),
        gr.components.Slider(10,80,label="Age (20-80)",step=1),
        gr.components.Slider(120,230,label="Height (Cm 120-230)",step=1),
        gr.components.Slider(35,135,label="Weight (Kg 35-135)")

    ],
    outputs=[gr.components.Number(label="Calories Burned")],
    title="Calories Prediction",

    description="Enter your information to predict calories burned during exercise.",
    css='div {margin-left: auto; margin-right: auto; width: 100%;\
            background-image: url("https://picsum.photos/id/213/1000/1500"); repeat 0 0;}')




# Start the Gradio app
iface.launch()
