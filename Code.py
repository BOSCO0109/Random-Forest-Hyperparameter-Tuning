from sklearn.model_selection import train_test_split , RandomizedSearchCV,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits , load_iris
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

file = load_digits()

x,y = file.data , file.target

rdm = RandomForestClassifier()

x_train , x_test , y_train ,y_test = train_test_split(x,y,random_state=42,test_size=0.25)

grid = {
    'n_estimators' :[5,10,50,100,200],
    'max_depth' : [None,1,5,10,20],
    'min_samples_leaf':[1,2,4],
    'min_samples_split':[2,5,10]
}

search = RandomizedSearchCV(rdm,param_distributions=grid,n_iter=10,cv=5,random_state=42)

search.fit(x_train,y_train)

y_pred =  search.predict(x_test)

AA = accuracy_score(y_test,y_pred=y_pred)

print(AA)
