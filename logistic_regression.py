import glob
from collections import Counter
from nltk.corpus import stopwords
import string
import numpy
import sys

ham_train = sys.argv[1]
spam_train = sys.argv[2]
ham_test = sys.argv[3]
spam_test = sys.argv[4]
Lamda = sys.argv[5]
iteration = sys.argv[6]
learning_rate = sys.argv[7]

# Training Data  Set
ham_train = glob.glob(ham_train+"/*.txt")
spam_train = glob.glob(spam_train+"/*.txt")

# Testing Data set
ham_test = glob.glob(ham_test+"/*.txt")
spam_test = glob.glob(spam_test+"/*.txt")

# the function reads the file and returns a counter of words for each file
def read_file(filename, stop_words, bayes):
    train_file = ""
    for file_names in filename:
        files = open(file_names)
        train_file += files.read()
        # remove for bayes
    #if bayes == 1:
    translator = str.maketrans('', '', string.punctuation)
    train_file = train_file.translate(translator)
    translator = str.maketrans('', '', string.digits)
    train_file = train_file.translate(translator)
    train_file = Counter(train_file.split())

    if stop_words:
        train_file = Counter([word for word in train_file if word not in stopwords.words('english')])
    return train_file

	ef slice_bag(bag_of_words,count):
    c=0
    for word in list(bag_of_words):
        if bag_of_words[word] < 2 and c <count:
            c+=1
            del bag_of_words[word]
    return bag_of_words

# the function produces the feature matrix with bag of words with count it happens in every file
def feature_matrix_(stop_word):

    # unique_train_ham = read_file(ham_train, stop_word, 1)
    unique_train_spam = read_file(spam_train, stop_word, 1)

    bag_of_words = unique_train_spam #+ unique_train_ham

    # print(bag_of_words)
    # bag_of_words = bag_of_words.most_common(104) # 10442

    bag_of_words = slice_bag(bag_of_words,count=4000)
    # print(len(bag_of_words))
    # print(bag_of_words)
    class_file = []
    feature_matrix = []

    for file_name in ham_train:
        feature = {}
        file = open(file_name)
        test_file = file.read()
        test_file = test_file.split(" ")
        # print(test_file)
        test_file =  Counter(test_file)
        # print(test_file)

        for words in bag_of_words:
            feature[words] =(test_file[words])
        # print(feature)
        feature["Probability_of_class"] = 0.0
        feature["Class_of_file"] = 0
        feature_matrix.append(feature)

    for file_name in spam_train:
        feature = {}
        file = open(file_name)
        test_file = file.read()
        test_file = test_file.split(" ")
        # print(test_file)
        test_file = Counter(test_file)
        # print(test_file)

        for words in bag_of_words:
            feature[words] = (test_file[words])
        feature["Probability_of_class"] = 0.0
        feature["Class_of_file"] = 1
        feature_matrix.append(feature)
    # print(feature_matrix)
    # print(len(bag_of_words))

    return feature_matrix, class_file, bag_of_words

# calculates the probability of the class given the weights and the feature matrix
def probability_of_class(w0,weights, feature_mat):
    count = 0

    for feature in feature_mat:
        sum_p = 0
        count+=1
        for words in weights:
            sum_p += weights[words] * feature[words]
        if sum_p < 700:
            prob = numpy.exp(numpy.array(w0 + sum_p, dtype=numpy.float)) /(1 + numpy.exp(numpy.array(w0 + sum_p,dtype=numpy.float)))
        else:
            prob = 1.0
        feature["Probability_of_class"] = prob

    return feature_mat

# function to updates the weight dependng on the learning rate and lambda
def update_weights(weights,n,lam,feature_matrix):
    for w in weights:
        sum = 0.0
        for feature in feature_matrix:
            sum += feature[w]*(feature["Class_of_file"] - feature["Probability_of_class"])
        weights[w] = weights[w] + n*sum - n*lam*weights[w]
    return weights

# Calculate sthe final accuracy of the classifier
def accuracy(weights):
    count_right =0;
    count_total =0;
    for file_name in spam_test:
        file = open(file_name)
        test_file = file.read()
        test_file = test_file.split(" ")
        test_file = Counter(test_file)
        sum_p = 0
        count_total += 1
        for words in weights:
            sum_p += weights[words] * test_file[words]
        if sum_p < 700:
            prob = numpy.exp(numpy.array(1 + sum_p, dtype=numpy.float)) /(1 + numpy.exp(numpy.array(1 + sum_p,dtype=numpy.float)))
        else:
            prob = 1.0
        if prob > 0.9:
            count_right += 1

    for file_name in ham_test:
        file = open(file_name)
        test_file = file.read()
        test_file = test_file.split(" ")
        test_file = Counter(test_file)
        sum_p = 0
        count_total += 1
        for words in weights:
            sum_p += weights[words] * test_file[words]
        if sum_p < 700:
            prob = numpy.exp(numpy.array(1 + sum_p, dtype=numpy.float)) /(1 + numpy.exp(numpy.array(1 + sum_p,dtype=numpy.float)))
        else:
            prob = 1.0
        if prob < 0.9:
            count_right += 1
    return count_right/count_total


def logistic_regression(iterations,n,lam,stop_word):
    global feature_matrix
    feature_matrix, class_file, bag_of_words = feature_matrix_(stop_word)
    w0 = 1.0
    initial_w = {}
    for word in bag_of_words:
        initial_w[word] = 1.0

    # print(feature_matrix)

    feature_matrix = probability_of_class(w0,initial_w,feature_matrix)

    w = initial_w
    for i in range(iterations):
        w = update_weights(w,n,lam,feature_matrix)
        feature_matrix = probability_of_class(w0, w, feature_matrix)
    

    final_accuracy = accuracy(w)

    if stop_word == 1:
        try:
            print("Logistic Regression - filtered - Total Accuracy - " + str(final_accuracy * 100))
        except ZeroDivisionError as err:
            print('Handling run-time error:', err)
    else:
        try:
            print("Logistic Regression - unfiltered - Total Accuracy - " + str(final_accuracy * 100))
        except ZeroDivisionError as err:
            print('Handling run-time error:', err)

logistic_regression(iterations=int(iteration), n=float(learning_rate), lam=float(Lamda), stop_word=0) # accuracy without removing stop-words
logistic_regression(iterations=int(iteration), n=float(learning_rate), lam=float(Lamda), stop_word=1) # accuracy with removing stop-words
