from ELM import *
from sklearn.datasets import fetch_mldata
from utils import *
import os

#Use only CPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

#Prepare MNIST dataset
mnist = fetch_mldata("MNIST original")
x, y = mnist.data / 255., mnist.target #Normalize data into [0, 1] range
indices = np.arange(60000)
num_train = 60000

#Shift values to be in [-1, 1] range
train_samples = x[indices[0:num_train]]*2-1
train_labels = to_categorical(y[indices[0:num_train]])*2-1
test_samples = x[60000:70000]*2-1
test_labels = to_categorical(y[60000:70000])*2-1

input_dim = 784
output_dim = 10
hidden_nodes = 1000
gamma = 1

#Creating ELM object
elm = ELM(input_dim, output_dim, hidden_nodes, gamma=1)

#Training
elm.train(train_samples, train_labels)

#Testing
train_acc = elm.evaluate(train_samples, train_labels)
test_acc = elm.evaluate(test_samples, test_labels)

#With 1,000 hidden nodes and gamma := 1, you should get around ~93% of training and testing accuracies
print("Training accuracy: {0}%".format(train_acc*100))
print("Testing accuracy: {0}%".format(test_acc*100))

