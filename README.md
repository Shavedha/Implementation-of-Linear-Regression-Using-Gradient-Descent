# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required Packages and read the .csv file
2. Define a function named ComputeCost and compute the output
3. Define a function named gradientDescent and iterate the loop
4. Predict the required graphs using scatterplots.

## Program:
```
Program to implement the linear regression using gradient descent.
Developed by: Y SHAVEDHA 
RegisterNumber:  212221230095
```
```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv("ex1.txt",header=None)

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def computeCost(x,y,theta):
    m=len(y) #length of training data
    h=x.dot(theta) #hypothesis
    square_err=(h-y)**2
    return 1/(2*m) * np.sum(square_err) #returning J
    
data_n=data.values
m=data_n[:,0].size
x=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
computeCost(x,y,theta)

def gradientDescent(x,y,theta,alpha,num_iters):
    m=len(y)
    J_history=[]
    for i in range(num_iters):
        predictions = x.dot(theta) #hypothesis
        error = np.dot(x.transpose(),(predictions -y))
        descent = alpha * 1/m * error
        theta-=descent
        J_history.append(computeCost(x,y,theta))
    return theta,J_history
 
theta,J_history  =gradientDescent(x,y,theta,0.01,1500)
print("h(x) = "+str(round(theta[0,0],2))+"+"+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="red")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def predict(x,theta):
    #takes in numpy array of x and theta and return the predicted value of y based on theta
    predictions=np.dot(theta.transpose(),x)
    return predictions[0]
    
predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0)))

```
## Output:
### Profit Prediction Graph
<img width="486" alt="image" src="https://user-images.githubusercontent.com/93427376/229060727-6ba29990-5291-4824-9a79-9c1061b60923.png">

### ComputeCost Value
<img width="272" alt="image" src="https://user-images.githubusercontent.com/93427376/229060998-83e543a9-de77-466f-80b2-ab934f9e369e.png">

### h(x) Value
<img width="300" alt="image" src="https://user-images.githubusercontent.com/93427376/229061218-26a2a95e-92d6-44fa-84b0-91af888a01f5.png">

### Cost Function using Gradient Descent
<img width="511" alt="image" src="https://user-images.githubusercontent.com/93427376/229061472-110e3793-6442-412b-927b-8ef15a0fd26a.png">

### Profit Prediction Graph
<img width="505" alt="image" src="https://user-images.githubusercontent.com/93427376/229061584-e6a7b089-4491-4caf-8c79-fd0b922ba076.png">

### Profit for the Population 35,000
<img width="535" alt="image" src="https://user-images.githubusercontent.com/93427376/229277350-51e9e8e6-639f-4193-888a-66113fc36b05.png">

### Profit for the Population 70,000
<img width="531" alt="image" src="https://user-images.githubusercontent.com/93427376/229277373-1500f063-b3d4-445f-9c7e-db25487775c9.png">

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
