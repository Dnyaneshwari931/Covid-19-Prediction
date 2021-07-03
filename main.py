import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
####### LOAD DATA
data = pd.read_csv('covid19india.csv')
data=data[['id','cases']]
print('-'*30);print('HEAD');print('-'*30)
print(data.head())

####### PREPARE DATA
print('-'*30);print('PREPARE DATA');print('-'*30)
x = np.array(data['id']).reshape(-1,1)#it shapes the array into 2D array with x original rows and one column so its(-1.1)
y = np.array(data['cases']).reshape(-1,1)
plt.plot(y,'-m')#plot the graph using y axis and x axis from 0-(n-1) with magenda and ----
#plt.show()

##### POLYNOMIAL FEATUREs
ployFeat = PolynomialFeatures(degree=9) # c + x1m + x2m for degree two
x = ployFeat.fit_transform(x) #takes x as on column and creates the other column of square in this case
#print(x)

#########TRAINING DATA
print('-'*30);print('TRAINING DATA');print('-'*30)
model = linear_model.LinearRegression()
model.fit(x,y)
accuracy = model.score(x,y)
print(f'Accuracy: {round(accuracy*100,3)} %')
y0 = model.predict(x)
plt.plot(y0, 'r*')
plt.show()

######PREDICTION
days=1
print('-'*30);print('PREDICTION');print('-'*30)
print(f'Prediction - Cases after {days} Days: ',end='')
print(round(int(model.predict(ployFeat.fit_transform([[514+days]])))/1000000,2),'Million')
