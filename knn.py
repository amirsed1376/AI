import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def jaygasht(number):
    jaygasht_list=[[]]
    help_list =[[]]
    for i in range(number):
        jaygasht_list[0].append("c")

    for i in range(number):
        help_list = jaygasht_list.copy()
        for j in help_list:
            j[number-i-1]="B"
            jaygasht_list.append(j.copy())
            j[number-i-1]="A"

    return (jaygasht_list)



dataset = pd.read_csv('train.csv')
x = dataset.iloc[ : , :].values
kmeans = KMeans(n_clusters= 5 ,random_state=0)
kmeans.fit(x)
labels = kmeans.labels_
print("centroids:")
for centroid in kmeans.cluster_centers_:
    print('centroid:', centroid)
print("_________________________")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(dataset , labels)
dataset = pd.read_csv("test.csv")
x = dataset.iloc[ : , :-1].values #all of rows without latest column
labels=knn.predict(x)
y = dataset.iloc[ : , -1:].values #all labls in test.csv (["A"] or ["B"]
"""this is for mapping ["A"] => "A" and ["B"]=>"B" """
test_labels = []
for i in y :
    test_labels.append(i[0])

max_accuracy=0
best_labeling=[]
"""for all states , calculate accuracy """

for jaygasht in jaygasht(5):
    current_prediction = []

    for h in labels:
        current_prediction.append(jaygasht[h])
    accuracy = accuracy_score(y_pred=current_prediction, y_true=test_labels)
    if accuracy >= max_accuracy:
        best_labeling=jaygasht
        max_accuracy=accuracy
    print(jaygasht)
    print(confusion_matrix(y_true=test_labels , y_pred=current_prediction ))
    print(accuracy_score(y_pred=current_prediction, y_true=test_labels))
    print("__________________")


print("**********************")
print("best_prediction:", best_labeling)
print("max_accuracy:",max_accuracy)





df = pd.DataFrame(x)
df[10] = list(labels)
# df = pd.DataFrame(x)
df.to_csv("a.csv")