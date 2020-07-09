import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
def svmclassifier():
    data = pd.read_csv("Winequality.csv")
    print(data.head())
    features= data.drop('quality',axis=1)
    # print(features)
    label = data['quality']
    # print(label)
    X_train, X_test, y_train, y_test = train_test_split(features,label,test_size=0.5)
    model = svm.SVC(kernel='linear')
    predict = model.fit(features,label)
    pred = model.predict(X_test)
    #print(pred)
    print(confusion_matrix(y_test,pred))
    print(classification_report(y_test,pred))
    print("The accuracy score is:",accuracy_score(y_test,pred))
if __name__=="__main__":
    svmclassifier()