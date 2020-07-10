import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.set_printoptions(precision=2)

dataset = pd.read_csv('./compton.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


#Preprocessing
from sklearn.model_selection import train_test_split
y = np.expand_dims(y, axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler(); sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train); y_train = sc_y.fit_transform(y_train)


#Linear Regression
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)

#Support Vector Regression
from sklearn.svm import SVR
svr_regressor = SVR(kernel = 'rbf')
svr_regressor.fit(X_train, y_train.ravel())

#Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
rf_regressor = RandomForestRegressor(n_estimators = 100)
rf_regressor.fit(X_train, y_train.ravel())

#Evaluating on Test Points
regressors = {'Linear Regression' : linear_regressor,
              'Support Vector Regression': svr_regressor,
              'Random Forest Regression': rf_regressor}
for name, regressor in regressors.items():
    y_hat = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_test)))
    rmse = np.sqrt(np.mean((y_test-y_hat)**2))
    tests = np.array([[12, 0.25, 2], [8, -1, 1]])
    tests_out = sc_y.inverse_transform(regressor.predict(sc_X.transform(tests)))
    print('Model = {}'.format(name))
    print('RMSE = {}'.format(rmse))
    print('Predicted {} for point {}, {}, {}.'.format(tests_out[0], tests[0,0], tests[0,1], tests[0,2]))
    print('Predicted {} for point {}, {}, {}.'.format(tests_out[1], tests[1, 0], tests[1, 1], tests[1, 2]))
    print()

#Plotting Predictions
E = [6 for i in range(100)]
T = np.linspace(-np.pi,np.pi, 100)
P = np.linspace(0, np.pi, 100)
for_plotting = sc_X.transform(np.array([E,T,P]).T)
linear_pred = sc_y.inverse_transform(linear_regressor.predict(for_plotting))
svr_pred = sc_y.inverse_transform(svr_regressor.predict(for_plotting))
rf_pred = sc_y.inverse_transform(rf_regressor.predict(for_plotting))
fig, ax = plt.subplots()
ax.plot(T, linear_pred, color='orange', label='Linear Regression')
ax.plot(T, svr_pred, color='red', label='SVR')
ax.plot(T, rf_pred, color='green', label='Random Forest')
ax.scatter(X_test[np.where(X_test[:,0] == 6.)][:,1], y_test[np.where(X_test[:,0]==6.)], label='Test Data')
ax.set_xlabel('$\\theta$', fontsize=14)
ax.set_ylabel('$\\frac{1}{\\sigma}\\frac{d\\sigma}{d\\Omega}$', fontsize=14)
plt.title('Scattering Distribution for a 6X Beam', fontsize=14)
plt.legend()
plt.savefig('regression.png',dpi=300)
plt.show()

