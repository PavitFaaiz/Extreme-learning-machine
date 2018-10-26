from include.data import get_data_set
import scipy.io as sio
import csv
from data_set.norb.smallnorb import *
from sklearn.datasets import fetch_mldata
import numpy as np
import tensorflow as tf

def to_categorical(labels):
    num_classes = np.max(labels).astype(np.int32)+1
    res = np.zeros([len(labels),num_classes])
    for i in range(len(labels)):
        label = (labels[i]).astype(np.int32)
        res[i][label] = 1

    return res

def load_dataset(dataset):
    if dataset == "MNIST":
        mnist = tf.keras.datasets.mnist
        (train_samples, train_labels), (test_samples, test_labels) = mnist.load_data()
        train_samples, test_samples = train_samples / 255.0, test_samples / 255.0
        train_labels = to_categorical(train_labels)
        test_labels = to_categorical(test_labels)

        num_train = 60000
        num_test = len(test_samples)
        input_shape = [1, 28, 28]
        input_size = np.prod(input_shape)

        train_samples = np.reshape(train_samples, [num_train, input_size])
        test_samples = np.reshape(test_samples, [num_test, input_size])

    elif dataset == "FASHION_MNIST":
        from tensorflow import keras
        fashion_mnist = keras.datasets.fashion_mnist
        (train_samples, train_labels), (test_samples, test_labels) = fashion_mnist.load_data()
        (train_samples, train_labels), (test_samples, test_labels)
        train_samples, test_samples = train_samples / 255.0, test_samples / 255.0
        train_samples, test_samples = train_samples / 255.0, test_samples / 255.0
        train_labels = to_categorical(train_labels)
        test_labels = to_categorical(test_labels)

        num_train = 60000
        num_test = len(test_samples)
        input_shape = [1,28,28]
        input_size = np.prod(input_shape)

        train_samples = np.reshape(train_samples, [num_train, input_size])
        test_samples = np.reshape(test_samples, [num_test, input_size])

    elif dataset == "CALTECH-SIL":
        caltech = sio.loadmat("caltech101_silhouettes_28_split1.mat")
        train_samples = caltech["train_data"]
        train_labels = caltech["train_labels"]-1
        test_samples = caltech["test_data"]
        test_labels = caltech["test_labels"]-1
        input_shape = [1,28,28]
        num_train = 4100

    elif dataset == "CIFAR-10":
        #get CIFAR-10
        train_samples, train_labels, train_l = get_data_set()
        test_samples, test_labels, test_l = get_data_set("test")
        train_samples = np.reshape(train_samples, [50000,32,32,3])
        train_samples = np.transpose(train_samples, [0, 3, 1, 2])
        train_samples = np.reshape(train_samples, [50000, 32*32*3])

        test_samples = np.reshape(test_samples, [10000,32, 32, 3])
        test_samples = np.transpose(test_samples, [0, 3, 1, 2])
        test_samples = np.reshape(test_samples, [10000, 32*32*3])
        train_labels = (train_labels)
        test_labels = (test_labels)
        input_shape = [3,32,32]
        num_train =50000

    elif dataset == "IRIS":
        f = open("data_set\\iris\\iris.data.csv")
        reader = csv.reader(f)
        data = np.zeros([150, 5])
        train_portion = 0.6
        num_train = int(150*train_portion)
        num_test = 150-num_train
        for i, row in enumerate(reader):
            for j, s in enumerate(row):
                data[i][j] = eval(s)

        train_samples = np.zeros([num_train, 4])
        train_labels = np.zeros([num_train, 1])
        test_samples = np.zeros([num_test, 4])
        test_labels = np.zeros([num_test, 1])

        for i in range(3):
            train_samples[i*num_train//3:(i+1)*num_train//3,0:4] = data[i*50:i*50+num_train//3,0:4]
            train_labels[i*num_train//3:(i+1)*num_train//3, 0] = data[i*50:i*50+num_train//3,4]
            test_samples[i * num_test//3:(i + 1) * num_test//3, 0:4] = data[(i+1) * 50 - num_test//3:(i+1) * 50, 0:4]
            test_labels[i * num_test//3:(i + 1) * num_test//3, 0] = data[(i+1) * 50 - num_test//3:(i+1) * 50, 4]

        train_labels = to_categorical(train_labels)
        test_labels = to_categorical(test_labels)
        input_shape = [4]

    elif dataset == "WINE":
        f = open("data_set\\wine.csv")
        reader = csv.reader(f)
        data = np.zeros([178, 14])
        train_portion = 0.6
        num_train = int(178 * train_portion)
        num_test = 178 - num_train
        for i, row in enumerate(reader):
            for j, s in enumerate(row):
                data[i][j] = eval(s)
        np.random.shuffle(data)
        train_samples = data[0:num_train, 1:]
        train_labels = data[0:num_train, 1]
        test_samples = data[num_train:, 1:]
        test_labels = data[num_train:, 1]
        train_labels = to_categorical(train_labels)
        test_labels = to_categorical(test_labels)
        input_shape = [13]

    elif dataset == "NORB":
        num_train = 24300
        num_test = 24300
        dataset = SmallNORBDataset(dataset_root="data_set\\norb")
        data_train = dataset.data['train']
        data_test = dataset.data['test']
        train_samples = np.zeros([num_train, 2*96*96])
        test_samples = np.zeros([num_test, 2*96*96])
        train_labels = np.zeros([num_train, 1])
        test_labels = np.zeros([num_test, 1])
        input_shape = [2, 96, 96]
        input_size = int(np.prod(input_shape))
        for s in range(len(data_train)):
            #Training
            #left image
            train_samples[s,0: input_size//2] = np.reshape(data_train[s].image_lt, [input_size//2])
            #right image
            train_samples[s, input_size//2: input_size] = np.reshape(data_train[s].image_rt, [input_size//2])
            train_labels[s] = data_train[s].category

            #Testing
            # left image
            test_samples[s, 0:input_size//2] = np.reshape(data_test[s].image_lt, [input_size // 2])
            # right image
            test_samples[s, input_size//2: input_size] = np.reshape(data_test[s].image_rt, [input_size // 2])
            test_labels[s] = data_test[s].category
        train_labels = to_categorical(train_labels)
        test_labels = to_categorical(test_labels)

    res = {}
    res["train_samples"] = train_samples
    res["train_labels"] = train_labels
    res["test_samples"] = test_samples
    res["test_labels"] = test_labels
    res["num_train"] = num_train
    res["num_test"] = len(test_labels)
    res["input_shape"] = input_shape
    res["num_classes"] = len(test_labels[0])
    return res

