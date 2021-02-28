import numpy as np
import matplotlib.pyplot as plt

input_size = 100 #4
output_size = 10 #3
hidden_size = 50 #5
num_examples = 1
# As Y = XW + b
# size of X is (data_num * data_size)
# size of W is (data_size * hidden_size)
# size of b has to be one-dimensional, equals with (hidden_size, 1)
# For First Initialize, set W, b by random number
# As we could set b by zero, my decision is that even if b sets random,
# performance of Neural Network would not be changed.

### INITILIZATION OF PARAMETERS ===================
W1 = np.random.randn(input_size, hidden_size)
b1 = np.random.randn(1, hidden_size)
W2 = np.random.randn(hidden_size, output_size)
b2 = np.random.randn(1, output_size)


### GENERATION OF DATA ============================
# Making data based on Y = 2X + 1
# target Y has some bias, from 0 ~ 1 distributed normal.
def make_data(num_examples, output_size):
    x = np.random.randn(num_examples, input_size)*5
    y = np.zeros((num_examples, output_size))

    for i in range(num_examples):
        for j in range(output_size):
            y[i][j] = 2*x[i, j] + 1 + np.random.normal()
    return x, y


### MODEL DEFINITION ============================

# We define two-layered model
def model(input):
    n1 = np.matmul(input, W1) + b1
    sig1 = 1/(1+np.exp(n1))

    n2 = np.matmul(sig1, W2) + b2
    sig2 = 1/(1+np.exp(n2))
    return sig1, sig2

### LOSS DEFINITION ============================

def loss_cross_entropy(y_t, y_pred, output_size):
    loss = 0
    for i in range(output_size):
        y_ti = y_t[:, i]
        y_predi = y_pred[:, i]

        loss_arr = y_ti*np.log(y_predi)-(1-y_ti)*np.log(1-y_predi)
        loss -= loss_arr
    return loss

### BACKPROPAGATION ===============================

def backpro(x, hidden_pred, input_size, output_size, hidden_size):
    grad = 0
    #y_t: (100, 10)
    # y_pred: (100, 10)
    for i in range(output_size): # 10
        for j in range(hidden_size): # 50
            y_ti = y_t[:, i]
            y_predi = y_pred[:, i]
            hidden_predj = hidden_pred[:, j]
            grad = (y_predi - y_ti)*W2[j, i]*hidden_predj*(1-hidden_predj)
            W2[:, j] -= grad

    grad = 0

    for i in range(output_size):
        for j in range(hidden_size):
            for k in range(input_size):
                y_ti = y_t[:, i]
                y_predi = y_pred[:, i]
                hidden_predj = hidden_pred[:, j]
                x_k = x[:, k]
                grad = (y_predi - y_ti)*W1[j, i]*hidden_predj*(1-hidden_predj)*x_k

            W1[:, j] -= grad



### MAIN ============================

if __name__ == "__main__":
    x, y_t = make_data(num_examples = num_examples, output_size = output_size)
    epoch = 0
    while(1):

        hidden_pred, y_pred = model(x)

        E = loss_cross_entropy(y_t, y_pred, output_size)
        backpro(x, hidden_pred, input_size, output_size, hidden_size)
        if(epoch%10==0):
            print(epoch, "   ", np.mean(E))
        if(epoch%20==0):
            plt.scatter(x[:, :output_size], y_t)
            plt.scatter(x[:, :output_size], y_pred)
            plt.scatter(x[:, :output_size], 2*x[:, :output_size] + 1)

            plt.show()
        epoch += 1
##VISUALIZE

#plt.scatter(x[:, :output_size], model(x))
#plt.scatter(x[:, :output_size], y)
#plt.scatter(x[:, :output_size], 2*x[:, :output_size] + 1)
#plt.show()

## ANALYSIS

