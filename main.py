import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def euclid_dist(a, b):
    # return dist and label
    return [np.linalg.norm(a - b[:-1]), b[-1]]


def min_dist(a, data, k):
    array = []
    for data_point in data:
        array.append(euclid_dist(a, data_point))
    arr = np.array(array)
    sorted_arr = arr[arr[:, 0].argsort()]
    minimum_k = []
    for i in range(0, k):
        minimum_k.append(sorted_arr[i, 1])
    return minimum_k


def knn_predict(a, data, k):
    minimum_k = min_dist(a, data, k)
    return (1, np.sum(minimum_k) / k) if np.sum(minimum_k) / k > 0.5 else (0, np.sum(minimum_k) / k)


def dtwoz():
    data = np.loadtxt("data/D2z.txt")
    X = []
    Y = []
    labels = []
    for i in range(-20, 20):
        x = i / 10.0
        for j in range(-20, 20):
            y = j / 10.0
            X.append(x)
            Y.append(y)
            labels.append(knn_predict([x, y], data, 1))

    pos = data[data[:, 2] == 1]
    neg = data[data[:, 2] == 0]
    fig, ax = plt.subplots()
    ax.scatter(X, Y, c=labels, s=10, cmap='bwr')
    ax.scatter(pos[:, 0], pos[:, 1], marker='+', facecolors='black')
    ax.scatter(neg[:, 0], neg[:, 1], marker='o', facecolors='none', edgecolors='black')
    plt.show()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def gradient(x, theta, y):
    return x * (sigmoid(np.dot(theta, x)) - y)


def logit_train(train_set, theta, lr):
    train_x = train_set[:, :-1]
    train_y = train_set[:, -1]
    for idx, x in enumerate(train_x):
        theta = theta - lr * gradient(x, theta, train_y[idx])
    return theta


def logit_predict(test_x, theta):
    return (1, sigmoid(np.dot(theta, test_x))) if sigmoid(np.dot(theta, test_x)) > 0.5 else \
        (0, sigmoid(np.dot(theta, test_x)))


def crossValid(df, test_idx, mode=1, k=1, lr=0.1):
    test_set = df.to_numpy()[test_idx:test_idx + 1000]
    train_set = np.vstack((df.to_numpy()[0:test_idx], df.to_numpy()[test_idx + 1000:]))

    test_data = test_set[:, 0:-1]
    test_y = test_set[:, -1]

    theta = np.zeros(test_data[0].size)
    if mode == 2:
        for e in range(0, 20):
            theta = logit_train(train_set, theta, lr)
            # print("e: {}, theta: {}".format(e, theta))
    np.savetxt("theta.txt", theta)

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    # For ROC curve
    confidence = []

    for idx, test_x in enumerate(test_data):
        pred = -1
        pred_prob = 0
        if mode == 1:
            prediction, prediction_prob = knn_predict(test_x, train_set, k)
            pred = prediction
            pred_prob = prediction_prob
        elif mode == 2:
            prediction, prediction_prob = logit_predict(test_x, theta)
            pred = prediction
            pred_prob = prediction_prob
        if pred == 1:
            if pred == int(test_y[idx]):
                confidence.append([pred_prob, True])
                tp += 1
            else:
                confidence.append([pred_prob, False])
                fp += 1
        elif pred == 0:
            if pred == int(test_y[idx]):
                tn += 1
            else:
                fn += 1

    # For ROC curve
    conf = np.array(confidence)
    sorted_conf = conf[(-conf[:, 0]).argsort()]
    correct = True
    tpr_list = [0]
    fpr_list = [0]
    tp_count = 0
    fp_count = 0
    auc = 0
    for instance in sorted_conf:
        if instance[1]:
            correct = True
            tp_count += 1
        else:
            if correct:
                h = (tp_count / tp) - tpr_list[-1]
                w = (1 / 2) * (fp_count / fp) + fpr_list[-1]
                auc += w * h
                tpr_list.append(tp_count / tp)
                fpr_list.append(fp_count / fp)
            correct = False
            fp_count += 1

    tpr_list.append(tp_count / tp)
    fpr_list.append(fp_count / fp)

    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print("{}NN Fold {} - Accuracy: {}, Precision: {}, Recall: {}".format(k, (test_idx + 1000) / 1000,
                                                                          accuracy, precision, recall))
    return accuracy, tpr_list, fpr_list, auc


def spamfilter(mode=1, k=1, lr=0.1, folds=5):
    df = pd.read_csv("data/emails.csv")
    df = df.iloc[:, 1:]
    tp_lists = []
    fp_lists = []
    # 5-fold cross validation
    sum_acc = 0
    auc = 0
    for i in range(0, folds * 1000, 1000):
        accuracy, tp_list, fp_list, auc = crossValid(df, i, mode, k, lr)
        sum_acc += accuracy
        tp_lists.append(tp_list)
        fp_lists.append(fp_list)
    average_acc = sum_acc / 5
    print("{}NN - Average Accuracy: {}".format(k, average_acc))
    return average_acc, tp_lists[0], fp_lists[0], auc


if __name__ == '__main__':
    # dtwoz()
    # # Q2.2, Q2.4
    # average_acc = []
    # k = [1, 3, 5, 7, 10]
    #
    # for i in k:
    #     av_acc, _, _, _ = spamfilter(mode=1, k=i, folds=5)
    #     average_acc.append(av_acc)
    # plt.plot(k, average_acc)
    # plt.xlabel("k")
    # plt.ylabel("Average Accuracy (5 Folds)")
    # plt.show()
    # Q2.3
    # spamfilter(mode=2, lr=0.001, folds=5)
    _, knn_tp_list, knn_fp_list, knn_auc = spamfilter(mode=1, k=5, folds=1)
    _, logit_tp_list, logit_fp_list, logit_auc = spamfilter(mode=2, lr=0.001, folds=1)
    plt.plot(knn_fp_list, knn_tp_list)
    plt.plot(logit_fp_list, logit_tp_list)
    plt.xlabel("False Positive Rate (Positive label: 1)")
    plt.ylabel("True Positive Rate (Positive label: 1)")
    plt.legend(loc='lower right', labels=["KNeighborsClassifier (AUC = {})".format(knn_auc),
                                          "LogisticRegression (AUC = {})".format(logit_auc)])
    plt.grid()
    plt.show()



