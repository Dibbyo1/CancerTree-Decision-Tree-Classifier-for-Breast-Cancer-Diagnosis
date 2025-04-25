from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

# Dataset: https://scikit-learn.org/stable/datasets/toy_dataset.html
X, y = datasets.load_breast_cancer(return_X_y=True)

print("There are", X.shape[0], "instances described by",
      X.shape[1], "features.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=42)

clf = tree.DecisionTreeClassifier(
    criterion='entropy', min_samples_split=6)  
clf = clf.fit(X_train, y_train) 

# Apply the decision tree to classify the data 'testData'.
predC = clf.predict(X_test) 

# Compute the accuracy of the classifier on 'testData'
print('The accuracy of the classifier is',
      accuracy_score(y_test, predC))  # (3 points)

# Visualize the tree created. Set the font size the 12 (5 points)
plt.figure(figsize=(40, 40))
_ = tree.plot_tree(clf, feature_names=datasets.load_breast_cancer(
).feature_names.tolist(), fontsize=12, filled=True)
plt.show()


# Visualize the training and test error as a function of the maximum depth of the decision tree
trainAccuracy = [] 
testAccuracy = [] 
depthOptions = range(1, 16) 
for depth in depthOptions: 
    cltree = tree.DecisionTreeClassifier(
        criterion='entropy', min_samples_split=6, max_depth=depth)
    # Decision tree training
    cltree = cltree.fit(X_train, y_train) 
    # Training error
    y_predTrain = cltree.predict(X_train) 
    # Testing error
    y_predTest = cltree.predict(X_test) 
    # Training accuracy
    trainAccuracy.append(accuracy_score(y_train, y_predTrain)) 
    # Testing accuracy
    testAccuracy.append(accuracy_score(y_test, y_predTest)) 

# Plot of training and test accuracies vs the tree depths 
plt.plot(depthOptions, trainAccuracy, marker='o', linestyle='-',
         color='blue', label='Training Accuracy') 
plt.plot(depthOptions, testAccuracy, marker='x', linestyle='-',
         color='green', label='Test Accuracy') 
plt.legend(['Training Accuracy', 'Test Accuracy'])
plt.xlabel('Tree Depth') 
plt.ylabel('Classifier Accuracy')

parameters = {'max_depth': range(1, 16)}  
clf = GridSearchCV(tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=6), parameters)
clf.fit(X_train, y_train) 
tree_model = clf.best_estimator_ 
print("The maximum depth of the tree should be", clf.best_params_['max_depth']) 

plt.figure(figsize=(40, 40))
_ = tree.plot_tree(tree_model, filled=True, fontsize=12)  # (5 points)
plt.show()
