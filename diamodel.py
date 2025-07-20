import pandas as pd
import joblib as jb

data = pd.read_csv(r"C:\Users\solan\OneDrive\Desktop\thonny\diabetes.csv")

#Data Analysis
print(data.head())
print(data.shape)
print(data.info())
print(data.columns)

x = data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = data[['Outcome']]
print(x)
print(y)

# STEP3- MODEL SELECTION
from sklearn.neighbors import KNeighborsClassifier#LinearRegression, Lasso, Ridge, KNeighborsClassifier#(nearest value or identified by your neighbourhood)
#model = LinearRegression()
#model = Lasso()
model = KNeighborsClassifier(n_neighbors=1)
print("Model Selected Successfully")

#STEP4- MODEL TRAINING

model.fit(x,y)
print("model trained")

# STEP5- RESULT
y_predicted= model.predict(x)
print("ML Predicted y : ",y_predicted)

# STEP6- MODEL ACCURACY
from sklearn.metrics import r2_score

acc = r2_score(y,y_predicted)
print("Model Performance : ",acc)

#max value of k

print(data['Outcome'].value_counts())


print(data['Outcome'].value_counts(normalize = True)*100)


# test the data through sample value

columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']

sample_data = pd.DataFrame({'Pregnancies':[6], 'Glucose':[148], 'BloodPressure':[72], 'SkinThickness':[35], 'Insulin':[0],
       'BMI':[33.6], 'DiabetesPedigreeFunction':[0.627], 'Age':[50]})
print("Outcome",model.predict(sample_data))

sample_data2 = pd.DataFrame([[1,85,66,29,0,26.6,0.351,31]], columns = columns)
print("Outcome",model.predict(sample_data2))

sample_data3 = pd.DataFrame([[0,89,12,36,0,26.6,2.451,66]], columns = columns)
print("Outcome",model.predict(sample_data3))


# model save
import joblib as jb
jb.dump(model,'Diabetes_KNN.pkl')
print("Model saved successfully")
