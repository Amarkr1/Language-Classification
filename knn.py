import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


def train(features_filename, values_filename):
	'''Load training data features and values.'''

	print("training...")
	sys.stdout.flush()

	features = np.asarray(pd.read_csv(features_filename).values.tolist())
	values = pd.read_csv(values_filename).as_matrix(columns=["Category"])

	return (features, values)


def knn(k, x, train_features, train_values):
	'''Find the most common value of the k-nearest neighbors of x.'''

	# find all Euclidean distances to x
	dists = np.linalg.norm(train_features - x, axis=1);

	# find k-nearest distances
	knearest_index = dists.argsort()[:k]
	knearest_dist = dists[knearest_index]
	knearest_lang = train_values[knearest_index].flatten()

	counts = np.bincount(knearest_lang)
	most_common = np.argmax(counts)

	return most_common


def predict(k, train_features, train_values, test_features):
	'''Predict value of each input feature using k-nn.'''

	print("predicting...")
	sys.stdout.flush()

	prediction = np.zeros((len(test_features), 2))
	#print(len(test_features))
	for i, x in enumerate(test_features):
		#print(i)
		lang = knn(k, x, train_features, train_values)
		prediction[i, 0] = i
		prediction[i, 1] = lang

	return prediction


def write_prediction(prediction, output_filename):
	'''Write prediction in "Id,Value" format to given output filename.'''

	output_file = open(output_filename, "w")
	output_file.write("Id,Category\n")

	for i, lang in prediction:
		output_file.write("%d,%d\n" % (i, lang))

	output_file.close()


def accuracy(prediction, actual_values):
	'''Calculate accuracy of the prediction.'''

	p = prediction[:, 1]
	a = actual_values[:, 0]

	return np.sum(prediction[:, 1] == actual_values[:, 0]) / float(len(p))


def test_best_k(train_features, train_values):
	'''Find accuracy of k-nn predictions with different values of k.'''

	test_k = range(1, 15)

	for k in test_k:
		# partition data into training and testing
		X_train, X_test, y_train, y_test = train_test_split(train_features, train_values, test_size=0.1)

		p = predict(k, X_train, y_train, X_test)
		acc = accuracy(p, y_test)
		print("k=%d, accuracy=%f\n" % (k, acc))


def cf(k, train_features, train_values):
	'''Generate confusion matrix.'''

	# partition data into training and testing
	X_train, X_test, y_train, y_test = train_test_split(train_features, train_values, test_size=0.1)
	p = predict(k, X_train, y_train, X_test)

	return confusion_matrix(y_test[:, 0], p[:, 1].astype(int))


if __name__ == "__main__":

	train_features_filename = "train_set_x_features.csv"
	train_values_filename = "train_set_y.csv"
	test_features_filename = "test_set_x_features.csv"
	test_values_filename = "test_set_y_knn.csv"

	# train
	train_features, train_values = train(train_features_filename, train_values_filename)

	# testing
	#test_best_k(train_features, train_values)

	k = 7

	# predict
	test_features = np.asarray(pd.read_csv(test_features_filename).values.tolist())
	prediction = predict(k, train_features, train_values, test_features)
	#print(prediction)
	write_prediction(prediction, test_values_filename)