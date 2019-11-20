#The data set contains information about money spent on advertisement and their generated sales. Money was spent on TV, radio and newspaper ads.

#import all the necessary libraries.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm

#Read the data using pandas.
data = pd.read_csv("data/Advertising.csv")
#To see what the data looks like
data.head()
data.columns
#As you can see, the column Unnamed: 0 is redundant. Hence, we remove it.
data.drop(['Unnamed: 0'], axis=1)

#Simple Linear Regression.
#Modelling
#For simple linear regression, let’s consider only the effect of TV ads on sales. Before jumping right into the modelling, let’s take a look at what the data looks like.
#We use matplotlib , a popular Python plotting library to make a scatter plot.
plt.figure(figsize=(16, 8))
plt.scatter(
    data['TV'],
    data['sales'],
    c='black'
)
plt.xlabel("Money spent on TV ads ($)")
plt.ylabel("Sales ($)")
plt.show()

#Let’s see how we can generate a linear approximation of this data.
X = data['TV'].values.reshape(-1,1)
y = data['sales'].values.reshape(-1,1)
reg = LinearRegression()
reg.fit(X, y)
print("The linear model is: Y = {:.5} + {:.5}X".format(reg.intercept_[0], reg.coef_[0][0]))

#Let’s visualize how the line fits the data.

predictions = reg.predict(X)
plt.figure(figsize=(16, 8))
plt.scatter(
    data['TV'],
    data['sales'],
    c='black'
)
plt.plot(
    data['TV'],
    predictions,
    c='blue',
    linewidth=2
)
plt.xlabel("Money spent on TV ads ($)")
plt.ylabel("Sales ($)")
plt.show()

#we need to look at the R² value and the p-value from each coefficient.
X = data['TV']
y = data['sales']
X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())

#Multiple Linear Regression.

Xs = data.drop(['sales', 'Unnamed: 0'], axis=1)
y = data['sales'].reshape(-1,1)
reg = LinearRegression()
reg.fit(Xs, y)
print("The linear model is: Y = {:.5} + {:.5}*TV + {:.5}*radio + {:.5}*newspaper".format(reg.intercept_[0], reg.coef_[0][0], reg.coef_[0][1], reg.coef_[0][2]))

# Let’s see by calculating the F-statistic, R² value and p-value for each coefficient.
X = np.column_stack((data['TV'], data['radio'], data['newspaper']))
y = data['sales']
X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())
