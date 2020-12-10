import pandas as pd
from flask import Flask, request, render_template
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))

dataset=pd.read_csv('Social_Network_Ads.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

def standardization(input):
    sc = StandardScaler()
    sc.fit(X_train)
    input = sc.transform(input)
    return input



@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    
    int_features = [int(x) for x in request.form.values()]
    
    
    
    prediction = model.predict(standardization([int_features]))
   
    if prediction==1:
      result="Purchased"
    else:
      result="Not purchased"
      
    return render_template('index.html',prediction_text=result)

if __name__=="__main__":
    app.run(debug=True)