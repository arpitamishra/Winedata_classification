import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
def naivebayesclassifier():
    data = pd.read_csv("Winequality.csv")
    print(data.head())
    features = data.drop('quality',axis=1)
    print(features)
    label = data['quality']
    print(label)
    CV = KFold(n_splits=100,random_state=42,shuffle=False)
    CV.get_n_splits(features,label)
    for train_index,test_index in CV.split(features):
        print("Train:",train_index,"Validation:",test_index)
        X_train,X_test = features[train_index],features[test_index]
        y_train,y_test = label[train_index],label[test_index]
    x_train,x_test,y_train,y_test = train_test_split(features,label,train_size=0.5)
    # print(x_test,x_train,y_train,y_test)
    model = MultinomialNB(alpha=1.0,fit_prior=True,class_prior=None)
    model.fit(x_train,y_train)
    prediction = model.predict(x_test)
    print(prediction)
    print("This is the confusion matrix:",confusion_matrix(y_test,prediction))
    print("This is the classification report:",classification_report(y_test,prediction))
    print("This is the accuracy:",accuracy_score(y_test,prediction))
    
if __name__=="__main__":
    naivebayesclassifier()