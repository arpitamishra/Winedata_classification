'''
Import libraries'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import v_measure_score
from sklearn.cluster import KMeans
def kmeansclustering():
    '''
    This is the kmeans clustering function'''
    data = pd.read_csv("Winequality.csv")
    print(data.head())
    features=data.drop('quality',axis=1)
    print(features.head())
    label = data['quality']
    print(label.head())
    x_train,x_test,y_train,y_test = train_test_split(features,label,train_size=0.8)
    # print(x_test)
    # print(x_train)
    # print(y_train)
    # print(y_test)
    kmeans = KMeans(n_clusters=11,random_state=0)
    kmeans.fit(x_train)
    print(kmeans.labels_)
    predict = kmeans.predict(x_test)
    #print(predict)
    print("The accuracy  is:",v_measure_score(y_test,predict))
    
if __name__=="__main__":
    kmeansclustering()
    