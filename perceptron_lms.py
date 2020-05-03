import numpy
import pandas
from sklearn.utils import shuffle
import time
from matplotlib import pyplot as plt
from numpy import genfromtxt

#lets load data set
#data format x1 x2 and y
train_data=pandas.read_csv("ML_HW2/train_set_d_1.txt",header=None, delimiter=r"\s+")
test_data=pandas.read_csv("ML_HW2/test_set_d_1.txt",header=None, delimiter=r"\s+")
rows=train_data.shape[0]
cols=train_data.shape[1]
 
print(rows)
print(cols)
#Lets check number of
 
print(train_data)
X=train_data.iloc[:,0:cols-1]  # input featurs x1 and x2
Y=train_data.iloc[:,cols-1] # label 1 or -1
Xtest=test_data.iloc[:,0:cols-1]
Ytest=test_data.iloc[:,cols-1]

no_of_weights=X.shape[1]
weight=numpy.zeros(no_of_weights+1)
learning_rate=0.01
start_time = time.time()
iterations=0
iters=[]
accuracy=[]

for epoch in range(0,10):
  
 
   for i in range(0,X.shape[0]):
       activation=weight[0]+sum(weight[1:]*(X.iloc[i,:]))
 
       #if activation>0:
           #prediction=1
       #else:
           #prediction=-1
 
       expected=Y[i]
       error=expected - activation
   
       weight[0]=weight[0]+(learning_rate*error)
       for j in range(1,3):
           weight[j]=weight[j]+(learning_rate*error*X.iloc[i,j-1])
      
       trues=0
       iters.append(iterations)
       iterations+=1
       
       for i in range(0,Y.shape[0]):
           activation=weight[0]+sum(weight[1:]*(X.iloc[i,:]))
 
           if activation>0:
               prediction=1
           else:
               prediction=-1

           expected=Y[i]
           if(expected==prediction):
               trues+=1
 
       print(trues)
       accuracy.append((trues/2000)*100)
       if(trues==2000):
           break
       print("------->")
  
   print(trues)
   if(trues==2000):
       print("stopped")
       print(iterations)
       break

print(weight)
 
trues=0
for i in range(0,Ytest.shape[0]):
   activation=weight[0]+sum(weight[1:]*(Xtest.iloc[i,:]))
 
   if activation>0:
       prediction=1
   else:
       prediction=-1
 
   expected=Ytest[i]
   if(expected==prediction):
       trues+=1


print(trues)

plt.figure()
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.plot(iters, accuracy, label="Accuracy (Learning rate = 0.01)")
plt.legend()
plt.show()