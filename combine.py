import pandas as pd
from numpy import poly1d
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def best_k_means(list_inertia):
    slope1_list = slope(list_inertia)
    slope2_list = slope(slope1_list)
    return slope2_list.index(min(slope2_list)) + 1


def slope(list_data):
    slope_list = []
    for i in range(0, len(list_data) - 1):
        tanx = abs((list_data[i + 1] - list_data[i]) / 10000)
        slope_list.append(tanx)

    return slope_list


def jaygashts(number):
    jaygasht_list = [[]]
    help_list = [[]]
    for i in range(number):
        jaygasht_list[0].append("c")

    for i in range(number):
        help_list = jaygasht_list.copy()
        for j in help_list:
            j[number - i - 1] = "B"
            jaygasht_list.append(j.copy())
            j[number - i - 1] = "A"

    return (jaygasht_list)


def get_test_lable_predict(k_knn, dataset_train, train_labels, x_test):
    """
     """
    knn = KNeighborsClassifier(n_neighbors=k_knn)
    knn.fit(dataset_train, train_labels)
    test_lable_predict = knn.predict(x_test)
    return test_lable_predict


def getInformationTest(file):
    """
        dataset of test
        make x_test = informations
        y_test = lables that is true  in test.csv
        :return x_test:data information
        :return y_test : labels of each x in test
    """
    dataset_test = pd.read_csv(file)
    x_test = dataset_test.iloc[:, :-1].values  # all of rows without latest column
    y_test = dataset_test.iloc[:, -1:].values  # all labls in test.csv (["A"] or ["B"]
    return x_test, y_test


def get_test_labels_true(y_test):
    """y_test is list of list
    this function became it to a list """
    test_labels_true = []
    for i in y_test:
        test_labels_true.append(i[0])
    return test_labels_true


def became_classLabel_to_label(label_classification, label_list):
    """became number to a label forexample 0=>A 1=>B ... """
    return_label_list = []
    for h in label_classification:
        return_label_list.append(label_list[h])
    return return_label_list


def get_label_dictionary(label, accuracy, confusionMatrix):
    label_dictionary = {}
    label_dictionary["label"] = label
    label_dictionary["accuracy"] = accuracy
    label_dictionary["confusionMatrix"] = confusionMatrix
    return label_dictionary


def get_knn_information_dictionary(labels_dict_list, max_current_accuracy, best_labeling , k_knn):
    knn_information_dictionary = {}
    knn_information_dictionary["k_knn"]=k_knn
    knn_information_dictionary["labels"] = labels_dict_list.copy()
    knn_information_dictionary["max_accuracy"] = max_current_accuracy
    knn_information_dictionary["best_labeling"] = best_labeling
    return knn_information_dictionary


def get_kmeans_information_dictionary(k_kmeans, train_labels, best_k_knn, max_accuracy_knn, knn_dict_list):
    kmeans_information_dictionary = {}
    kmeans_information_dictionary["k_kmeans"] = k_kmeans
    kmeans_information_dictionary["train_labels"] = train_labels
    kmeans_information_dictionary["centers"] = kmeans.cluster_centers_
    kmeans_information_dictionary["best_k_knn"] = best_k_knn
    kmeans_information_dictionary["max_accuracy_knn"] = max_accuracy_knn
    kmeans_information_dictionary["knn_dict_list"] = knn_dict_list
    return kmeans_information_dictionary


def information_best_classification_file(directory_name, kmeans_dict, x_train ,inetia_list):
    """write information of best classifcation in to a file """
    file_name = directory_name + "information.txt"
    cvs_file = directory_name + "train_classification.csv"
    file = open(file_name, "w+")
    best_k_knn = kmeans_dict["best_k_knn"]
    best_labeling=kmeans_dict["knn_dict_list"][best_k_knn[0]]["best_labeling"]
    file.write("best number classification = " + str(kmeans_dict["k_kmeans"]) + "\n")
    file.write("inertia="+str(inetia_list[kmeans_dict["k_kmeans"]])+"\n")
    # file.write("amount of error = " + "\n")  # TODO amount of error
    file.write("best_k_knns = " + str(best_k_knn) + "\n")
    file.write("max accuracy ="+ str(kmeans_dict["max_accuracy_knn"]) + "\n")
    file.write("best labeling = "+str(best_labeling) + "\n")
    file.write("centers =\n")
    for center in (kmeans_dict["centers"]):
        file.write(str(center) + "\n")
    file.write("labels file =" + cvs_file + "\n")
    df = pd.DataFrame(x_train)
    df["class"] = list(kmeans_dict["train_labels"])
    df["label"]=list(became_classLabel_to_label(label_classification=kmeans_dict["train_labels"], label_list=best_labeling))
    df.to_csv(cvs_file)

def make_files_for_information_KMeans(kmeans_list  ,directory_file):
    for kmeans_dict in kmeans_list:
        best_k_knn = kmeans_dict["best_k_knn"]
        best_labeling = kmeans_dict["knn_dict_list"][best_k_knn[0]]["best_labeling"]
        file = open(directory_file+"k_kmeans" + str(kmeans_dict["k_kmeans"]) +".txt" , "w+")
        file.write("number classification = " + str(kmeans_dict["k_kmeans"]) + "\n")
        file.write("best_k_knns = " + str(best_k_knn) + "\n")
        file.write("max accuracy =" + str(kmeans_dict["max_accuracy_knn"]) + "\n")
        file.write("best labeling = " + str(best_labeling) + "\n")
        file.write("centers =\n")
        for center in (kmeans_dict["centers"]):
            file.write(str(center) + "\n")
        for knn_dict in kmeans_dict["knn_dict_list"]:
            file.write("###################################\n")
            file.write( "if k_knn = " + str(knn_dict["k_knn"]) + "\n")
            file.write("max accuracy =" + str(knn_dict["max_accuracy"]) + "\n")
            file.write("best labeling = " + str(knn_dict["best_labeling"]) + "\n")
            for label_dict in knn_dict["labels"]:
                file.write("___________________\n")
                file.write("if labeling = "+str(label_dict["label"])+"\n")
                file.write("confusion matrix ="+str(label_dict["confusionMatrix"][0])+"\n")
                file.write("                  "+str(label_dict["confusionMatrix"][1])+"\n")
                file.write("accuracy="+str(label_dict["accuracy"])+"\n")

def make_csv_test(kmeans_dict , directory_file , x_test , x_train , y_test):
    cvs_file = directory_file +"test_labeling.csv"
    df =pd.DataFrame(x_test)
    df["y"]=get_test_labels_true(y_test=y_test)
    train_predict_list = []
    for knn_dict in (kmeans_dict["knn_dict_list"]):
        test_lable_predict=get_test_lable_predict(k_knn=knn_dict["k_knn"] ,dataset_train=x_train ,train_labels=kmeans_dict["train_labels"] ,x_test=x_test)
        labeling= became_classLabel_to_label(label_classification=test_lable_predict , label_list=knn_dict["best_labeling"])
        df["k"+str(knn_dict["k_knn"])]=list(labeling)
    df.to_csv(cvs_file)


def predict_train_file(x_train,train_labels ,label_list , file ):
    file2 = open("informations/best_classification/train_confusion_matrix", "w+")
    data=pd.read_csv(file)
    x= data.iloc[:, :].values
    df = pd.DataFrame(x)

    for k_knn in range(1,20):
        # train_labels=train_labels[0:3000]
        predict_label = get_test_lable_predict(k_knn=k_knn, dataset_train=x_train, train_labels=train_labels,x_test=x_train)
        y_train = became_classLabel_to_label(label_classification=predict_label, label_list=label_list)
        y_train = get_test_labels_true(y_test=y_train)
        accuracy=accuracy_score(y_true=became_classLabel_to_label(label_classification=train_labels ,label_list=label_list) , y_pred=y_train)
        matrix=confusion_matrix(y_true=became_classLabel_to_label(label_classification=train_labels ,label_list=label_list) , y_pred=y_train)
        df["k"+str(k_knn)]=y_train
        df.to_csv(file)
        file2.write("\n___________________\n")
        file2.write("if k_knn ="+str(k_knn)+"\n")
        file2.write("accuracy=\n"+str(accuracy))
        file2.write("\nmatrix=\n"+str(matrix))




x_test, y_test = getInformationTest("test.csv")
informatio_list = []
inetia_list = []
dataset_train = pd.read_csv('train.csv')

x_train = dataset_train.iloc[:, :].values
k_means_dict_list = []
for k_kmeans in range(1, 10):
    knn_dict_list = []
    kmeans = KMeans(n_clusters=k_kmeans, random_state=0)
    kmeans.fit(x_train)
    inetia_list.append(kmeans.inertia_)
    train_labels = kmeans.labels_


    max_accuracy_knn = 0
    best_k_knn = [0]

    for k_knn in range(1, 20):
        print("k_KMeans = " , k_kmeans , "    k_KNN = " , k_knn)
        labels_dict_list = []

        test_lable_predict = get_test_lable_predict(k_knn=k_knn, dataset_train=dataset_train, train_labels=train_labels,
                                                    x_test=x_test)

        test_labels_true = get_test_labels_true(y_test=y_test)

        max_current_accuracy = 0
        best_labeling = []
        """for all states , calculate accuracy """
        for label in jaygashts(k_kmeans):
            current_prediction = became_classLabel_to_label(label_classification=test_lable_predict, label_list=label)
            accuracy = accuracy_score(y_pred=current_prediction, y_true=test_labels_true)
            confusionMatrix = confusion_matrix(y_pred=current_prediction, y_true=test_labels_true)

            if accuracy >= max_current_accuracy:
                best_labeling = label
                max_current_accuracy = accuracy



            """fill label dictionary """

            label_dictionary = get_label_dictionary(label=label, accuracy=accuracy, confusionMatrix=confusionMatrix)
            labels_dict_list.append(label_dictionary.copy())

        if max_current_accuracy == max_accuracy_knn:
            best_k_knn.append(k_knn)

        if max_current_accuracy > max_accuracy_knn:
            """ compare current accuracy in current k_knn  with best_k_knn"""
            best_k_knn = [k_knn]
            max_accuracy_knn = max_current_accuracy

        knn_information_dictionary = get_knn_information_dictionary(labels_dict_list=labels_dict_list,
                                                                    max_current_accuracy=max_current_accuracy,
                                                                    best_labeling=best_labeling , k_knn=k_knn)
        knn_dict_list.append(knn_information_dictionary.copy())



    kmeans_information_dictionary = get_kmeans_information_dictionary(k_kmeans=k_kmeans, train_labels=train_labels,
                                                                      best_k_knn=best_k_knn,
                                                                      max_accuracy_knn=max_accuracy_knn,
                                                                      knn_dict_list=knn_dict_list)
    k_means_dict_list.append(kmeans_information_dictionary.copy())


plt.plot(range(1, len(inetia_list) + 1), inetia_list)
plt.savefig("plot.png")
best_kMeans = best_k_means(list_inertia=inetia_list)
print("best k for k_means is ", best_kMeans)

information_best_classification_file(directory_name="informations/best_classification/",
                                     kmeans_dict=k_means_dict_list[best_kMeans-1], x_train=x_train , inetia_list=inetia_list)



make_files_for_information_KMeans(kmeans_list=k_means_dict_list , directory_file="informations/")

make_csv_test(kmeans_dict=k_means_dict_list[best_kMeans-1] ,directory_file="informations/best_classification/" ,x_test=x_test ,x_train=x_train , y_test=y_test)
best_label = k_means_dict_list[best_kMeans-1]["knn_dict_list"][best_kMeans]["best_labeling"]
predict_train_file(x_train=x_train , train_labels=k_means_dict_list[best_kMeans-1]["train_labels"] , label_list=best_label ,file="informations/best_classification/train_classification.csv")