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

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

data=fetch_california_housing()
X=data.data[:, :3]
Y=np.column_stack((data.target, data.data[:, 6]))

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

scaler_X=StandardScaler()
scaler_Y=StandardScaler()

X_train=scaler_X.fit_transform(X_train)
X_test=scaler_X.transform(X_test)
Y_train=scaler_Y.fit_transform(Y_train)
Y_test=scaler_Y.transform(Y_test)

sgd=SGDRegressor(max_iter=1000,tol=1e-3)
multi_output_sgd=MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train,Y_train)
Y_pred=multi_output_sgd.predict(X_test)

Y_pred=scaler_Y.inverse_transform(Y_pred)
Y_test=scaler_Y.inverse_transform(Y_test)
mse=mean_squared_error(Y_test,Y_pred)
print("Mean Squared Error:",mse)
print("\nPredictions:\n",Y_pred[:5])
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
