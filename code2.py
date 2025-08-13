#using searchgridcv we tunning the data into more accuracy 


file = datasets.load_digits()
x,y = file.data , file.target

x_train , x_test , y_train , y_test = train_test_split(x,y,random_state=42,test_size=0.25)

svm = SVC()

grid = {'C':[0.1,1,10,100,1000],'kernel':['linear','poly','rbf'],'gamma':['scale','auto',1,0.1,0.01,0.001]}

search = GridSearchCV(svm,cv=5,param_grid=grid,scoring='accuracy')

search.fit(x_train,y_train)

y_pred = search.predict(x_test)

AA = accuracy_score(y_test,y_pred=y_pred)

print(AA)


output : 988888889
