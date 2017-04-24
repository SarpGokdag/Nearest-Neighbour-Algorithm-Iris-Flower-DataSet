from sklearn import datasets
iris = datasets.load_iris()
x = iris.data
y = iris.target
from sklearn.cross_validation import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = .2)
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()
classifier.fit(xTrain, yTrain)
tahminler = classifier.predict(xTest)
from sklearn.metrics import  accuracy_score
print(accuracy_score(yTest, tahminler))
