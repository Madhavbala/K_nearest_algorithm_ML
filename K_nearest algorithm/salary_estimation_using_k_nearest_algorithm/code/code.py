#Importing Libraries
import pandas as pd #useful for loading the dataset
import numpy as np #to perform array
import matplotlib.pyplot as plt
from tqdm import tqdm
# Choose Dataset from Local Directory


#Load Dataset

dataset = pd.read_csv('C:/Users/madhavan.bala/Documents\ML\Day - 2 Salary estimation using K-Nearest Neighbour/Dataset/salary.csv')
print(dataset.shape)
print(dataset.head(5))

#Mapping Salary Data to Binary Value

income_set = set(dataset['income'])
dataset['income'] = dataset['income'].map({'<=50K': 0, '>50K': 1}).astype(int)
print(dataset.head(5)

#Segregate Dataset into X(Input/IndependentVariable) & Y(Output/DependentVariable)

inpu = dataset.iloc[:, :-1].values
outp = dataset.iloc[:, -1].values


#Splitting Dataset into Train & Test

from sklearn.model_selection import train_test_split
inpu_train, inpu_test, outp_train, outp_test = train_test_split(inpu, outp, test_size = 0.25, random_state = 0)



from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
inpu_train = sc.fit_transform(inpu_train) 
inpu_test = sc.transform(inpu_test)

#Finding the Best K-Value

error = []
from sklearn.neighbors import KNeighborsClassifier
# Calculating error for K values between 1 and 40
for i in range(1,40):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(inpu_train, outp_train)
    pred_i = model.predict(inpu_test)
    error.append(np.mean(pred_i != outp_test))

print ("Model    :   ",model)

plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()

#Training
#minkowski is a formula 
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors = 2, metric = 'minkowski', p = 2)
model.fit(inpu_train, outp_train)


# *Prediction for all Test Data
outp_pred = model.predict(inpu_test)
print(np.concatenate((outp_pred.reshape(len(outp_pred),1), outp_test.reshape(len(outp_test),1)),1))

# *Evaluating Model - CONFUSION MATRIX 

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(outp_test, outp_pred)

print("Confusion Matrix: ")
print(cm)

print("Accuracy of the Model: {0}%".format(accuracy_score(outp_test, outp_pred)*100))

#get the new employee input to predit
name = input("Enter Employee Name : ")
age = int(input("Enter New Employee's Age: "))
edu = int(input("Enter New Employee's Education: "))
cg = int(input("Enter New Employee's Captital Gain: "))
wh = int(input("Enter New Employee's Hour's Per week: "))
newEmp = [[age,edu,cg,wh]]
result = model.predict(sc.transform(newEmp))
print(result)

if result == 1:
  print(f"{name}  Employee Eligible to get above 50K")
else:
  print(f"{name}  Employee not eligible to get Salary above 50K")
