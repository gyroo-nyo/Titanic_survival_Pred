import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("Titanic-Dataset.csv")
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer.fit(df.iloc[:,5:].values)
df.iloc[:,5:] = imputer.transform(df.iloc[:,5:].values)
df["Sex"]=df["Sex"].map({"male":0, "female":1})


model = RandomForestClassifier(n_estimators=100, random_state=42)
y=df["Survived"].values
X=df[["Pclass","Sex","Fare",]].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 10)
model.fit(X_train,y_train)
result=model.predict(X_test)
accuracy=accuracy_score(y_test,result)

scores=cross_val_score(model,X_train,y_train,cv=5)
train_score=model.score(X_train,y_train)
test_score=model.score(X_test,y_test)

param={"n_estimators":[50,100,200],"max_depth":[None,10,20],"min_samples_split":[2,5,10]}

grid=GridSearchCV(estimator=model,param_grid=param,scoring='accuracy',cv=5)
grid.fit(X_train,y_train)


best_model=grid.best_estimator_
k=best_model.score(X_train,y_train)
j=best_model.score(X_test,y_test)

import matplotlib.pyplot as plt

y_pred = best_model.predict(X_test)


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Survived", "Survived"])
disp.plot()
plt.title("Confusion Matrix")
plt.show()

print(classification_report(y_test, y_pred))
from joblib import dump,load
dump(model,"titanic_survival.joblib")
loaded_model = load("titanic_survival.joblib")