import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import csv

# Download dataset and split to train and test sets
dataset = pandas.read_csv("datasets/Titanic_train.csv")
eval_set = pandas.read_csv("datasets/Titanic_test.csv")  # set without target values
dataset, target = dataset.drop("Survived", axis=1), dataset['Survived'].copy()
X_train, X_test, y_train, y_test = train_test_split(dataset, target, test_size=0.2, random_state=42)


# Modify data
def data_preparation(X):
    X["Has_cabin"] = X["Cabin"].apply(lambda x: 1 if type(x) == str else 0)
    X['Embarked'] = X['Embarked'].fillna('S')
    X['Fare'] = X['Fare'].fillna(X['Fare'].median())
    X['Age'] = X['Age'].fillna(X['Age'].median())
    X["Sex"] = X["Sex"].apply(lambda x: 1 if x == "male" else 0)
    X["Name_length"] = X["Name"].apply(len)
    X['Embarked'] = X['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
    X = X.drop(drop_elements, axis=1)

    return X


X_train = data_preparation(X_train.copy())
X_test = data_preparation(X_test.copy())


# Function, that returns a list of models
def models_list():
    rf = RandomForestClassifier(n_estimators=1000, max_leaf_nodes=16, n_jobs=-1, random_state=42)
    ex_tree = ExtraTreesClassifier(random_state=42)
    svc = SVC(probability=True)
    ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), learning_rate=0.5)
    gboost = GradientBoostingClassifier(max_depth=2, n_estimators=3)

    return [("rf", rf), ("ex_tree", ex_tree), ("svc", svc), ("ada", ada), ("gboost", gboost)]


# Using soft voting classifier and grid_search
voting_clf = VotingClassifier(estimators=models_list(), voting='soft')
voting_clf.fit(X_train, y_train)
y_pred = voting_clf.predict(X_test)

print("Soft voting accuracy: ", accuracy_score(y_test, y_pred))

# Run model on test set and save to csv file
eval_set_modified = data_preparation(eval_set.copy())
y_pred = voting_clf.predict(eval_set_modified)

f = open('datasets/submit.csv', 'w')
writer = csv.writer(f)
writer.writerow(["PassengerId", "Survived"])

for i, j in zip(list(eval_set['PassengerId']), list(y_pred)):
    writer.writerow([i, j])

f.close()
