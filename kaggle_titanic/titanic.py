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
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

def main():
	    # my code here
	# Read all the data files. 
	# train.csv contains all the training data. test.csv will be used for evaluating the models using various 
	# scores -- accuracy score, confusion matrix and classification report. test.csv contains all the passenger
	# info (X_test), gender_submission.csv contains the survival result (Y_test).  
	url = "data/train.csv"
	test_url = "data/test.csv"
	survive_url = "data/gender_submission.csv"

	names = [ 'PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked' ];
	test_names = [ 'PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked' ];
	survive_names = [ 'PassengerId' , 'Survived'];

	dataset = pandas.read_csv(url, names=names)[1:]
	testset = pandas.read_csv(test_url, names=test_names)[1:]
	surviveset = pandas.read_csv(survive_url, names=survive_names)[1:]
	
	# Preprocess. This will change data format from string to integers, and remove any NaN's.
	dataset = preprocess(dataset);
	testset = preprocess(testset);
	
	# Remove unwanted fields from X's like PassengerId and Names. 
	# Standardize data.
	X_train = dataset[[ 'Pclass','Sex', 'Age', 'SibSp','Parch','Fare','Embarked' ]];
	Y_train = dataset['Survived'];
	scaler = StandardScaler().fit(X_train);
	X_train = scaler.transform(X_train);
	
	X_test = testset[[ 'Pclass','Sex', 'Age', 'SibSp','Parch','Fare','Embarked' ]];
	Y_test = surviveset['Survived'];
	scaler = StandardScaler().fit(X_test);
	X_test = scaler.transform(X_test);
	
	seed = 7;
	scoring = 'accuracy';

	models = []
	models.append(('LR', LogisticRegression()))
	models.append(('LDA', LinearDiscriminantAnalysis()))
	models.append(('KNN', KNeighborsClassifier()))
	models.append(('CART', DecisionTreeClassifier()))
	models.append(('NB', GaussianNB()))
	models.append(('SVM', SVC()))
	# evaluate each model in turn
	results = []
	names = []
	for name, model in models:
		kfold = model_selection.KFold(n_splits=10, random_state=seed)
		cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
		results.append(cv_results)
		names.append(name)
		print name 
		msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
		print(msg)

		model.fit(X_train, Y_train);
		prediction = model.predict(X_test);
		print accuracy_score(Y_test, prediction);
		print confusion_matrix(Y_test, prediction);
		print classification_report(Y_test, prediction);

# Scrub the Nan's and put in the data median.
def scrub_nan(dataset):
	dataset[np.isnan(dataset.astype(float))] = np.median(dataset[~np.isnan(dataset.astype(float))].astype(float));
	return dataset;

def preprocess(dataset):
	
	le = preprocessing.LabelEncoder();
	le.fit(dataset['Sex']);
	dataset['Sex'] =  le.transform(dataset['Sex']);

	le.fit(dataset['Embarked']);
	dataset['Embarked'] =  le.transform(dataset['Embarked']);

	print dataset['Age'];
	dataset['Age'] = scrub_nan(dataset['Age']);
	dataset['Fare'] = scrub_nan(dataset['Fare']);

	return dataset

if __name__ == "__main__":
	main()

