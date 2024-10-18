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
![Screenshot 2024-10-18 112729](https://github.com/user-attachments/assets/c11b0c2c-4686-4cc4-9ea3-b2cca87488b6)
![Screenshot 2024-10-18 112735](https://github.com/user-attachments/assets/394bd6e0-5c70-4ace-a6a4-52a9b11082aa)
![Screenshot 2024-10-18 112740](https://github.com/user-attachments/assets/6de0ea47-df8b-47ef-87b8-c458fac8f0e1)
![Screenshot 2024-10-18 112750](https://github.com/user-attachments/assets/ce00ca3a-f387-472c-8a5e-46d51af1a5c7)
![Screenshot 2024-10-18 112755](https://github.com/user-attachments/assets/6909cf37-3284-4ef6-86d8-cbb4383050eb)
![Screenshot 2024-10-18 112802](https://github.com/user-attachments/assets/ad6792b5-4e2f-41ad-9721-b00f706195b2)
![Screenshot 2024-10-18 112809](https://github.com/user-attachments/assets/692053dc-023b-4d03-a030-540972ce587e)






## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
