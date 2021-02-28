import numpy as np

hidden_size = 4

# Reference
# http://jaejunyoo.blogspot.com/2017/01/backpropagation.html

# As Y = XW + b
# size of X is (data_num * data_size)
# size of W is (data_size * hidden_size)
# size of b has to be one-dimensional, equals with (hidden_size, 1)
# For First Initialize, set W, b by random number
# As we could set b by zero, my decision is that even if b sets random,
# performance of Neural Network would not be changed.


### GENERATION OF DATA ============================

# Making data based on Y = 2X + 1
# target Y has some bias, from 0 ~ 1 distributed normal.
def make_data():
    data = []
    for x1 in range(2):
        for x2 in range(2):
            data.append([x1, x2])
    data = np.asarray(data, dtype = np.int32)
    """
    input1 input2 | label 
    [0]      [0]  |  [1]
    [0]      [1]  |  [0]
    [1]      [0]  |  [0]
    [1]      [1]  |  [1] 

    """
#    label = np.logical_xor(data[:, 0], data[:, 1])
    label = np.asarray([0, 1, 1, 0])
    return data, label


### MODEL DEFINITION ============================

# We define two-layered model
class model(object):
    def __init__(self):
        self.input = np.random.normal(loc = 0.1, scale = 0.05, size = (2, hidden_size))
        self.hidden = np.random.normal(loc = 0.1, scale = 0.05, size = (hidden_size, 1))

        self.z1 = None
        self.z2 = None
        self.wz1 = None
        self.wz2 = None

    @staticmethod
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    def forward(self, x):
        self.wz1 = np.matmul(x, self.input)
        z1 = self.sigmoid(self.wz1)
        self.z1 = z1

        self.wz2 = np.matmul(z1, self.hidden)
        z2 = self.sigmoid(self.wz2)
        self.z2 = z2

        return self.z2


    ### BACKPROPAGATION ===============================

    def backward(self, y, y_pred, x):
        y = y.reshape(-1, 1)
        grad_hidden = (self.hidden * (y_pred - y))
        self.hidden += 0.01 * grad_hidden

#        grad_input = np.matmul((y_pred - y) * self.out * self.hidden * (1 - self.hidden), x)
        for k in range(2):
            for i in range(1):
                self.input[k, :] += np.matmul((y_pred - y)*self.wz2[:, i]*self.z1*(1-self.z1), x[:, k])


### MAIN ============================

if __name__ == "__main__":
    model = model()
    data, label = make_data()
    for _ in range(1):
        y_hat = model.forward(data)
        model.backward(label, y_hat, data)
    print("training     1 time, accuracy equals\n" , model.forward(data))
    for _ in range(100):
        y_hat = model.forward(data)
        model.backward(label, y_hat, data)
    print("training  5000 time, accuracy equals\n", model.forward(data))
    for _ in range(10000):
        y_hat = model.forward(data)
        model.backward(label, y_hat, data)
    print("training 10000 time, accuracy equals\n", model.forward(data))


