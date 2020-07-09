import pandas as pd
from sklearn.model_selection  import train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import confusion_matrix,classification_report,v_measure_score
def ridgeclassifier():
    data = pd.read_csv("winequality.csv")
    print(data.head())
    features = data.drop("quality",axis=1)
    print(features)
    label = data['quality']
    print(label)
    x_train,x_test,y_train,y_test = train_test_split(features,label,train_size=0.5)
    print(x_train,x_test,y_train,y_test)
    model = RidgeClassifier()
    model.fit(x_train,y_train)
    pred = model.predict(x_test)
    # print(pred)
    print("classification report is :",classification_report(y_test,pred))
    print("confusion matrix is:",confusion_matrix(y_test,pred))

if __name__=="__main__":
    ridgeclassifier()
