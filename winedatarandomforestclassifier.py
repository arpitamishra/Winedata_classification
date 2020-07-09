import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
def randomforestclassifier():
    data = pd.read_csv("Winequality.csv")
    print(data.head())
    features=data.drop('quality',axis=1)
    print(features.head())
    label = data['quality']
    print(label.head())
    x_train,x_test,y_train,y_test = train_test_split(features,label,train_size=0.5)
    print(x_test)
    print(x_train)
    print(y_train)
    print(y_test)
    model = RandomForestClassifier(n_estimators=10,criterion='gini',max_features=7,min_samples_leaf=1,random_state=44,min_samples_split=2)
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    #print(y_pred)
    print("Accuracy:",accuracy_score(y_test, y_pred))
    # print(model.feature_importances_)
    importances = pd.Series(model.feature_importances_,index=features.columns)
    indices = np.argsort(importances)[::-1]
    print("Feature ranking:")
    for f in range(features.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(features.shape[1]), importances[indices],
           color="r", align="center")
    plt.xticks(range(features.shape[1]), indices)
    plt.xlim([-1, features.shape[1]])
    plt.show()

if __name__=="__main__":
    randomforestclassifier()
    