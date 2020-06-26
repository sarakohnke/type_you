#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 12:06:46 2020

@author: sarakohnke
"""


#Find feature importances in model
import pandas as pd
feat_importances = pd.Series(clf_rf2.feature_importances_, index=X_rf2.columns)
feat_importances.to_csv('feat_importances.csv')
feat_importances.nlargest(10).plot(kind='barh')

#test model against dummy regressor
import numpy as np
predicted_rf2 = clf_rf2.predict(X_test_rf2)

#my model plot
plt.scatter(y_test_rf2,predicted_rf2)
plt.xlabel('Actual values')
plt.ylabel('Predicted values')

plt.plot(np.unique(y_test_rf2), np.poly1d(np.polyfit(y_test_rf2, predicted_rf2, 1))(np.unique(y_test_rf2)))

#plt.savefig('r2.png',dpi=300)

from sklearn.dummy import DummyRegressor
import numpy as np
import matplotlib.pyplot as plt
dummy_mean=DummyRegressor(strategy='mean')
predicted_rf2 = clf_rf2.predict(X_test_rf2)
predicted_dummy=dummy_mean.predict(X_test_rf2)

plt.scatter(y_test_rf2,predicted_dummy)
plt.xlabel('Actual values')
plt.ylabel('Predicted values')

plt.plot(np.unique(y_test_rf2), np.poly1d(np.polyfit(y_test_rf2, predicted_dummy, 1))(np.unique(y_test_rf2)))
#plt.show()