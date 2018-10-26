from ELM import *
from sklearn.datasets import fetch_mldata
from utils import *
from data_set .load_dataset import *
import os

#Use only CPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

#Prepare dataset
'''
    Dataset names
    - MNIST
    - FASHION_MNIST
    - CIFAR-10
    - CALTECH-SIL
    - IRIS
    - NORB
    - WINE
'''
data = load_dataset("CIFAR-10")
train_samples = data["train_samples"]
train_labels = data["train_labels"]
test_samples = data["test_samples"]
test_labels = data["test_labels"]
num_train = data["num_train"]
num_test = len(test_labels)
input_shape = data["input_shape"]
num_classes = len(test_labels[0])

#Creating ELM object
hidden_nodes = 1000
input_dim = np.prod(input_shape)
output_dim = num_classes
elm = ELM(input_dim, output_dim, hidden_nodes, gamma=1)

#Training
elm.train(train_samples, train_labels)

#Testing
train_acc = elm.evaluate(train_samples, train_labels)
test_acc = elm.evaluate(test_samples, test_labels)

#With 1,000 hidden nodes and gamma := 1, you should get around ~93% of training and testing accuracies
print("Training accuracy: {0}%".format(train_acc*100))
print("Testing accuracy: {0}%".format(test_acc*100))

