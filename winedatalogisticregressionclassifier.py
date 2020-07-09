import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
def logisticregressionclassifier():
    data=pd.read_csv("Winequality.csv")
    print(data.head())
    features = data.drop('quality',axis=1)
    print(features.head())
    label = data['quality']
    print(label.head())
    x_train,y_train,x_test,y_test=train_test_split(features,label,train_size=0.5)
    print(x_train,x_test,y_train,y_test)
    model = LogisticRegression(multi_class='multinomial',solver='newton-cg')
    model.fit(x_train,y_train)
    pred = model.predict(x_test)
    print(pred)
    # print("The confusion matrix is:",confusion_matrix(y_test,pred))
    # print("The classification report is:",classification_report(y_test,pred))
    # print("The accuracy is:",accuracy_score(y_test,pred))
if __name__=="__main__":
    logisticregressionclassifier()