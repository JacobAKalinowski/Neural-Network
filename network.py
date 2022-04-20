
#Import the necessary packages

import numpy as np
import matplotlib.pyplot as plt

#First build an abstract Layer class that will be used for linear and 
#activation layers
#Both layers will contain forward and backward functions

class Layer():
    def __init__(self):
        #Only shared variable is input
        self.input = None
        
    def forward(self, input_data):
        raise NotImplementedError()
        
    def backward(self, error, lr):
        raise NotImplementedError()

class Linear(Layer):
    def __init__(self, input_size, output_size):
        super(Linear, self).__init__()
        self.weights = np.random.rand(input_size,output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5
        
    def forward(self, input_data):
        self.input = input_data
        out = np.dot(self.input, self.weights)+self.bias
        return out
    
    def backward(self, error, lr):
        input_error = np.dot(error, self.weights.T)
        weight_error = np.dot(self.input.T, error)
        
        self.weights -= lr*weight_error
        self.bias -= lr*error
        
        return input_error

class Activation(Layer):
    def __init__(self, act, act_prime):
        super(Activation, self).__init__()
        self.activation = act
        self.activation_prime = act_prime
    
    def forward(self, input_data):
        self.input = input_data
        return self.activation(self.input)
    
    def backward(self, error, lr):
        error = self.activation_prime(self.input)*error
        return error

class Network():
    def __init__(self, loss, loss_prime):
        self.layers = []
        self.loss = loss
        self.loss_prime = loss_prime
        self.errors = []
        self.epochs = None
        
    def add_layer(self, layer):
        #Can add activation or linear layer here
        self.layers.append(layer)
        
    def predict(self, input_data):
        input_size = len(input_data)
        prediction = []
        #Forward Propagation
        for i in range(input_size):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward(output)
            prediction.append(output)
        
        return prediction
    
    def train(self, x, y, epochs, lr):
        sample_size = len(x)
        self.epochs = epochs
        
        for epoch in range(epochs):
            error = 0
            for j in range(sample_size):
                #Forward
                output = x[j]
                for layer in self.layers:
                    output = layer.forward(output)
                
                #Update error
                error += self.loss(output, y[j])
                running_error = self.loss_prime(output, y[j])
                
                #Backward
                for layer in reversed(self.layers):
                    running_error = layer.backward(running_error, lr)
                    
            error /= sample_size
            self.errors.append(error)
            print("Epoch %d/%d, Error = %f" %(epoch+1, epochs, error))
        
    def plot_error(self, title):
        #Method to plot errors after training
        x = np.linspace(1, self.epochs, self.epochs)
        y = self.errors
        plt.plot(x, y)
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.show()

#Some activation functions

#ReLU
def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    x[x>0] = 1
    x[x<=0] = 0
    return x

#Sigmoid
def sigmoid(x):
    e = 1/(1+np.exp(-x))
    return e/e.sum(axis = 0)

def sigmoid_prime(x):
    return np.exp(-x)/((1+np.exp(-x))**2)

#tanh
def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1-np.tanh(x)**2

#Loss Function
def mse(y_pred, y):
    return np.mean(np.power(y_pred - y, 2))

def mse_prime(y_pred, y):
    return 2*(y_pred - y)/float(y.size)

from keras.datasets import mnist
from keras.utils import np_utils

# load MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshape and normalize input data, encode output data
x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
x_train = x_train.astype('float32')
x_train /= 255
y_train = np_utils.to_categorical(y_train)

# same for test data
x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
x_test = x_test.astype('float32')
x_test /= 255
y_test = np_utils.to_categorical(y_test)

# Network
MyNet = Network(mse, mse_prime)
MyNet.add_layer(Linear(28*28, 100))               
MyNet.add_layer(Activation(tanh, tanh_prime))
MyNet.add_layer(Linear(100, 100))                 
MyNet.add_layer(Activation(tanh, tanh_prime))
MyNet.add_layer(Linear(100, 50))                 
MyNet.add_layer(Activation(tanh, tanh_prime))
MyNet.add_layer(Linear(50, 10))                    
MyNet.add_layer(Activation(tanh, tanh_prime))

# train on 1000 samples
MyNet.train(x_train[0:1000], y_train[0:1000], epochs=200, lr=0.09)

# test on 3 samples
out = MyNet.predict(x_test[0:3])
print("\n")
print("predicted values : ")
print(out, end="\n")
print("true values : ")
print(y_test[0:3])

MyNet.plot_error('Error Rate for MNIST, lr = 0.09')
