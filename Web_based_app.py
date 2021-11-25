# Load the trained best model
import pickle
filename = 'finalized_model.sav'
loadmodel = pickle.load(open(filename, 'rb'))

import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load dataset for data normalization purpose 
df = pd.read_csv('dataset.csv')
dummies_Gender = pd.get_dummies(df['Gender'])
dummies_Geo = pd.get_dummies(df['Geography'])
dummies_Exited = pd.get_dummies(df['Exited'])
df=df.join(dummies_Gender)
df=df.join(dummies_Geo)
df=df.join(dummies_Exited)
df=df.drop(['Gender','Geography','F','Quebec','No','Exited'],axis=1)
Y=df['Yes']
X=df.drop(['Yes'],axis=1)
train_x, test_x, train_y, test_y = train_test_split(X,Y, train_size=0.8)
sc_X = MinMaxScaler()
train_x = sc_X.fit_transform(train_x)
test_x = sc_X.fit_transform(test_x)

# Web GUI
from flask import Flask, request, render_template
app = Flask(__name__)

@app.route("/")
def index():
    return render_template('loyaltypred.html') #This html is saved under the template folder

@app.route("/send",methods=['POST'])
def send(sum=sum): #function for "Predict" button
    try:
        if request.method=='POST':
            credit=request.form['credit']
            age=request.form['age']
            tenure=request.form['tenure']
            balance=request.form['balance']
            numofpro=request.form['numofpro']
            hascr=request.form['hascr']
            isact=request.form['isact']
            salary=request.form['salary']
            gender=request.form['gender']
            geo=request.form['geography']
            if credit=='' or credit=='' or age=='' and tenure=='' or balance=='' or salary=='': #user must enter all entries
                return render_template('loyaltypred.html',sum="Please enter all entries before prediction")
            if gender=='Male':
                M='1'
            else:
                M='0'
            if gender=='Male':
                M='1'
            else:
                M='0'
            if geo=='Alberta':
                Alberta='1'
                Manitoba='0'
            elif geo=="Manitoba":
                Alberta='0'
                Manitoba='1'
            else:
                Alberta='0'
                Manitoba='0'
            if hascr=='Yes':
                hascr='1'
            else:
                hascr='0'
            if isact=='Yes':
                isact='1'
            else:
                isact='0'
            
            a=np.expand_dims([credit,age,tenure,balance,numofpro,hascr,isact,salary,M,Alberta,Manitoba],axis=0) 
            b=loadmodel.predict(sc_X.transform(a))
            if b==1:
                sum='Yes'
            else:
                sum='No'
       
            return render_template('loyaltypred.html',sum=sum)
    except:
        return render_template('loyaltypred.html',sum="Please enter numerical entries properly.") # Catch errors if entries are not numerical

if __name__ == "__main__":
    app.run(debug='True')