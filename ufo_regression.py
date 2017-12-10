from sklearn.linear_model import Ridge
from sklearn import preprocessing
from sklearn.feature_selection import f_regression
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats import outliers_influence
from yellowbrick.regressor import ResidualsPlot
from regressors.stats import residuals
from math import sqrt
import seaborn as sns

from util import *

sightings = pd.read_csv("scrubbed.csv", 
	names=['datetime', 'city', 'state', 'country', 'shape',
            'duration (seconds)', 'duration (hours/min)', 'comments',
            'date posted', 'latitude', 'longitude'],
        usecols=['datetime', 'shape', 'duration (seconds)', 'date posted', 'latitude', 'longitude'],
        dtype={'datetime' : str, 'shape' : 'category', 'duration (seconds)' : str, 
            'latitude' : str, 'longitude' : str},
        header=0)

#PREPROCESSING
#Data sanitization and basic formatting
sightings['shape'] = sightings['shape'].str.lower()
sightings['shape'] = sightings['shape'].astype('category')
sightings['duration (seconds)'] = pd.to_numeric(sightings['duration (seconds)'], errors='coerse', downcast='float')
sightings['latitude'] = pd.to_numeric(sightings['latitude'], errors='coerce', downcast='float')
sightings['longitude'] = pd.to_numeric(sightings['longitude'], errors='coerce', downcast='float')
sightings = sightings.dropna()

#Group longitude and latitude
"""
Y: Lat min: -82.8628 max: 72.7
X: Long min: -176.658 max: 178.442
Using 32 secotrs, 8 along latitude, 4 along longitude
"""
sightings['latitude'] = sightings['latitude'].apply(prepLatCoord)
sightings['longitude'] = sightings['longitude'].apply(prepLongCoord)
coordSec = sightings['latitude'].add(sightings['longitude'])
coordSec = coordSec.astype(int)
coordSec = coordSec.apply(coordSector)

"""
#Coordinate representing sectors, used for generating heatmap
coordDF = pd.DataFrame(index=range(8), columns=range(4))
coordDF = coordDF.fillna(0)
for sec in coordSec.iteritems():
        y = sec[1] // 4
        x = sec[1] - 4 * y
        coordDF[x][y] += 1;
sns.heatmap(coordDF, cmap='gist_heat')
plt.show()
"""

#Datetime conversion
sightings['datetime'] = sightings['datetime'].apply(dtToWd)

#Location and shape dummy encoding
shapeDV = pd.get_dummies(sightings['shape'])
coordDV = pd.get_dummies(coordSec, prefix="sector")

regressionParams = pd.concat(objs=[sightings['datetime'], coordDV, shapeDV], axis=1)
regressionParams.dropna()

#REGRESSION
#Ridge regression helps reduce multicollinearity and does not assume normality
clf = Ridge()
results = clf.fit(regressionParams, sightings['duration (seconds)'])
score = results.score(regressionParams, sightings['duration (seconds)'])


#Standardized residual plot
"""
#Calculate the standardized residuals
predicted = clf.predict(regressionParams)
predicted = pd.Series(predicted)
residuals = predicted.subtract(sightings['duration (seconds)'])
mse = mean_squared_error(sightings['duration (seconds)'], predicted)
rmse = sqrt(mse)
stdResid = residuals.apply(func=(lambda x : x / rmse))
stdResid.plot.hist(bins=100, range=[-0.2, 0.2])
plt.show()
"""

"""
#Correlation matrix heatmap
corr = regressionParams.corr()
sns.heatmap(corr)
plt.show()
"""

#OUTPUT

#Descriptive statistics output
"""
print(sightings['datetime'].describe())
print(coordDV.describe(include='all'))
print(shapeDV.describe(include='all'))
print(sightings['duration (seconds)'].describe())
"""

"""
#Output table
outModel = sm.OLS(sightings['duration (seconds)'], regressionParams)
outModelFU = outModel.fit()
outResFR = outModel.fit_regularized(alpha=1.0, L1_wt=0)
outRes = sm.regression.linear_model.OLSResults(outModel, 
        outResFR.params,
        outModel.normalized_cov_params)
outOLSInfl = outliers_influence.OLSInfluence(outRes)
"""

"""
Full MSE: 356336749975
Reduced SSR: 2.79340477734e+16
Full SSR: 2.79354e+16
Partial F: 0.0882
"""

print(score)
#Coefficients in order: datetime, coord sector, shape
#print(list(coordDV))
#print(list(shapeDV))
#print(results.coef_)
#print(results.intercept_)
#print(f_regression(regressionParams, sightings['duration (seconds)'], center=False))
print("Num sectors: " + str(len(coordDV.columns)) + " Num unique shapes: " + str(len(shapeDV.columns)))
