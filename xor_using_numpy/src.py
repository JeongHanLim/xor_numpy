import numpy as np

hidden_size = 4


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
    label = np.logical_xor(data[:, 0], data[:, 1])
    return data, label


### MODEL DEFINITION ============================

# We define two-layered model
class model(object):
    def __init__(self):
        self.input = np.random.normal(loc = 0.1, scale = 0.05, size = (2, hidden_size))
        self.hidden = np.random.normal(loc = 0.1, scale = 0.05, size = (hidden_size, hidden_size))
        self.out = np.random.normal(loc = 0.1, scale = 0.05, size = (hidden_size, 1))

        self.z1 = None
        self.z2 = None

    @staticmethod
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    def forward(self, x):
        z1 = np.matmul(x, self.input)
        z1 = self.sigmoid(z1)
        self.z1 = z1
        z2 = np.matmul(z1, self.hidden)
        z2 = self.sigmoid(z2)
        self.z2 = z2
        z3 = np.matmul(z2, self.out)
        y = self.sigmoid(z3)

        return y

    ### LOSS DEFINITION ============================
    """
    def loss_cross_entropy(y_t, y_pred, output_size):
        loss = 0
        for i in range(output_size):
            y_ti = y_t[:, i]
            y_predi = y_pred[:, i]

            loss_arr = y_ti*np.log(y_predi)-(1-y_ti)*np.log(1-y_predi)
            loss -= loss_arr
        return loss
    """

    ### BACKPROPAGATION ===============================

    def backward(self, y, y_pred, x):
        y = y.reshape(-1, 1)
        if ((y_pred - y).any != 0 or y_pred.any != 0 or (1 - y_pred).any != 0):
            grad_out = (y_pred - y) / (y_pred * (1 - y_pred))
            # print(y.shape)
            # print((y_pred-y).shape)
            # print(grad_out.shape)
            # print(self.out.shape)
            self.out += 0.01 * grad_out

            grad_hidden = np.matmul(self.hidden, (y_pred - y))
            self.hidden += 0.01 * grad_hidden

            #        print(((y_pred-y)*self.out*self.hidden*(1-self.hidden)).shape)
            #        print(x.shape)
            #        print(self.input.shape)
            grad_input = np.matmul((y_pred - y) * self.out * self.hidden * (1 - self.hidden), x)
            grad_input = grad_input.reshape(2, 4)
            self.input += 0.01 * grad_input


### MAIN ============================

if __name__ == "__main__":
    model = model()
    data, label = make_data()
    for _ in range(10000):
        y_hat = model.forward(data)
        model.backward(label, y_hat, data)
    print(model.forward(data))
