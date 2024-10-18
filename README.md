# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import necessary libraries
2. Load dataset from 'Placement_Data.csv'
3. Create a copy of the dataset
4. Drop unnecessary columns
5. Check for missing values and duplicates
6. Encode categorical variables
7. Set features X and target y
8. Split data into training and testing sets
9. Create Logistic Regression model
10. Fit model on training data
11. Predict on test data
12. Calculate accuracy and generate confusion matrix
13. Predict for a specific input
14. Print evaluation results

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SANJAI.R
RegisterNumber:  212223040180
*/
```
```py


import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
![Screenshot 2024-10-18 112729](https://github.com/user-attachments/assets/6c324bd2-f115-4e54-91f0-bbace9ae739d)
![Screenshot 2024-10-18 112735](https://github.com/user-attachments/assets/a0720257-3342-4217-9e49-ed68c6a5e326)
![Screenshot 2024-10-18 112740](https://github.com/user-attachments/assets/850235d3-6906-4a47-a497-1f91dda3df66)
![Screenshot 2024-10-18 112750](https://github.com/user-attachments/assets/b9859a80-3aba-45c9-ba9b-9e83b0354206)
![Screenshot 2024-10-18 112755](https://github.com/user-attachments/assets/ac28a63b-db41-47cc-913a-623900a662b1)
![Screenshot 2024-10-18 112802](https://github.com/user-attachments/assets/f8a410ad-6ab7-4d81-8e69-3d33a003135e)
![Screenshot 2024-10-18 112809](https://github.com/user-attachments/assets/c69b46c2-e441-4435-935e-8f4379ea0784)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
