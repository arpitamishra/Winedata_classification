#loading required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,classification_report,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
def knnfunction():
    #reading data
    data = pd.read_csv("Winequality.csv")
    print(data.head())#displaying first five rows of the dataset.
    #extracting fetaures and labels
    features = data.drop("quality",axis =1)
    print("The features are ",features)
    quality = data['quality']
    print("The quality values are",quality)
    #splitting dataset into training set and test set
    X_test,X_train,Y_test,Y_train = train_test_split(features,quality,test_size=0.5)
    print(X_test)
    print(X_train)
    print(Y_test)
    print(Y_train)
    #fitting model
    knn = KNeighborsClassifier(n_neighbors=6)
    knn.fit(features,quality)
    #prediction
    predicted = knn.predict(X_test)
    print(predicted)
    #evaluation metrics
    print(confusion_matrix(Y_test,predicted))
    print(classification_report(Y_test,predicted,labels=[5,6,7]))
    print(accuracy_score(Y_test,predicted))
if __name__=="__main__":
    knnfunction()
