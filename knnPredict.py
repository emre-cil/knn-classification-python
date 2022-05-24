# Step 1 - Load Data
import pandas as pd
dataset = pd.read_csv("purchase_logs.csv") # provide full path of the dataset between quotes
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 3].values

# Step 2 - Convert Gender to number
from sklearn.preprocessing import LabelEncoder
labelEncoder_gender =  LabelEncoder()
X[:,0] = labelEncoder_gender.fit_transform(X[:,0])

# Step 3 - Split into training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Step 4 - Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#plot and output data
array_accuracy = []
array_precision = []
array_recall = []

#function train and returns knn classifier according to euclidean.
def get_train(i):
    clsf = KNeighborsClassifier(n_neighbors=i, metric='minkowski', p=2)
    clsf.fit(X_train, y_train)
    return clsf
# prediction part
from sklearn import metrics

for i in range(1,60):
    y_predX = get_train(i).predict(X_test)
    cm = metrics.confusion_matrix(y_test, y_predX) 
    accuracy = metrics.accuracy_score(y_test, y_predX) 
    precision = metrics.precision_score(y_test, y_predX) 
    recall = metrics.recall_score(y_test, y_predX) 
    array_accuracy.append(accuracy)
    array_precision.append(precision)
    array_recall.append(recall)
    print("Accuracy score:",accuracy)
    print("Precision score:",precision)
    print("Recall score:",recall)
    print("Confusion matrix: \n", cm)
    print("\n")

#plot part
import matplotlib.pyplot as plt
plt.plot(array_accuracy, label="accuracy")
plt.plot(array_precision, label="precision")
plt.plot(array_recall, label="recall")
plt.legend()
plt.xlabel('neighbors')
plt.ylabel('score')
plt.title('Accuracy, Precision and Recall figure')
plt.show()

