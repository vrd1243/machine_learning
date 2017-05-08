import numpy as np
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.metrics import mean_squared_error
import scipy

def rmse_cv(model, X, y):
    rmse= np.sqrt(-model_selection.cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)

def main():

	train_url = "data/train.csv"
	test_url = "data/test.csv"
	test_y_url = "data/sample_submission.csv"

	# Fetch all the data from files
	train = pandas.read_csv(train_url);
	test  = pandas.read_csv(test_url);
	test_y  = pandas.read_csv(test_y_url);
	data = pandas.concat((train.loc[:,'MSSubClass':'SaleCondition'], test.loc[:,'MSSubClass':'SaleCondition']));

	# Prediction feature is adjusted to log scale. 
	train['SalePrice'] = np.log1p(train['SalePrice']);
	
	# Determine all numeric features and if their skew is high, adjust to log scale.
	numeric_feats = data.dtypes[data.dtypes != "object"].index;
	skewed_feats = train[numeric_feats].apply(lambda x: scipy.stats.skew(x.dropna()));
        skewed_feats  = (skewed_feats[skewed_feats > 0.75]).index;
	data[skewed_feats] = np.log1p(data[skewed_feats]);

	# Handle the non-numeric features. There are two ways to handle this. 
	# One way is to use the get_dummies and create fake additional features based on the values of these
	# non-numeric features. The second way is to use LabelEncoder().
	# Observations : With the get_dummies() approach, the linear regression reports very high 
	# prediction error.

	# data = pandas.get_dummies(data);
	le = preprocessing.LabelEncoder();
	object_feats = data.dtypes[data.dtypes == "object"].index;
	data[object_feats] = data[object_feats].apply(preprocessing.LabelEncoder().fit_transform);
	
	# Remove the NaNs.
	data = data.fillna(data.mean());
	
	# Get the training and test data.
	X_train = data[:train.shape[0]]
	X_test = data[train.shape[0]:]
	Y_train = train.SalePrice
	Y_test = np.asarray(np.log1p(test_y.SalePrice));

	# Lasso and Linear Regression models.
	model_linear = LinearRegression();
	model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]);
	lasso = rmse_cv(model_lasso, X_train, Y_train);
	linear      = rmse_cv(model_linear, X_train, Y_train);
	print lasso;
	print linear;
	
	# Perform fitting on the training data.  
	models = [];
	models.append(model_linear);
	models.append(model_lasso);
	
	#  For each model, perform fitting on the training data, and perform prediction on test data.
	for model in models:
		model.fit(X_train, Y_train);
		Y_pred = model.predict(X_test);
		print mean_squared_error(np.expm1(Y_test), np.expm1(Y_pred));

if __name__ == "__main__":
	main()

