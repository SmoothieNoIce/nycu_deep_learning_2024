# %% 
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', category=Warning)
# acitvation function sigmoid
def sigmoid(x):
    return 1 / (1+np.exp(-x))

def derivative_sigmoid(x):
    return sigmoid(x) * (1-sigmoid(x))

# acitvation function tanh
def tanh(x):
    res = (np.exp(x) - np.exp(-x) ) / 2
    return res

def derivative_tanh(x):
    t1 = np.ones([x.shape[0], 1])
    t2 = np.square(tanh(x))
    res =  t1 - t2
    return res

# acitvation function relu
def relu(x):
    res = np.maximum(0, x)
    return np.maximum(0, x)

def derivative_relu(x):
    res = np.where(x > 0, 1, 0)
    return res
    
# Loss function Mean Square Error
def mse_loss(pred_y, y):
    return np.mean((pred_y-y)**2)

# Drivative of Mean Square Error
def derivative_mse_loss(pred_y, y):
    return 2 * (pred_y-y) / y.shape[0]

# Loss function Cross Entropy
def cross_entropy_loss(pred_y, y):
    test = 1
    t1  = pred_y*np.log(y)
    t2 = (test-pred_y)
    t3  = np.log(test-y)
    return np.mean( t1 + t2 * t3 )

# Drivative of Cross Entropy
def derivative_cross_entropy_loss(pred_y, y):
    test = np.ones([pred_y.shape[0], 1])
    return ((pred_y-y)/(y * (test - y))) / y.shape[0]


class Layer():
    def __init__(
            self, in_features, out_features, bias=True, activate=True):
        self.bias = bias
        self.activate = activate
        self.w = np.random.normal(0, 1, size=(in_features, out_features))
        if self.bias:
            self.b = np.zeros(out_features)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.local_grad = x
        self.z = np.matmul(x, self.w)
        if self.bias:
            self.local_grad_b = np.ones((x.shape[0], 1))
            self.z += self.b
        out = self.z
        if self.activate:
            out = sigmoid(out)
        return out

    def backward(self, prev_wT_delta):
        self.up_grad = prev_wT_delta # 前一層的計算結果，Ex:  最後一為 loss 的微分
        if self.activate:
            self.up_grad *= derivative_sigmoid(self.z) # z 為 x 和 w 的變化量
        res = np.matmul(self.up_grad, self.w.T)
        return res

    def step(self, lr):
        grad_w = np.matmul(self.local_grad.T, self.up_grad)  # Multiply its output delta and input activation to get the gradient of the weight
        self.w -= lr * grad_w # Subtract a ratio (percentage) of the gradient from the weight
        if self.bias:
            grad_b = np.matmul(self.local_grad_b.T, self.up_grad)
            self.b -= lr * grad_b.squeeze()


class Network():
    def __init__(
            self, num_features, lr=0.001, bias=False, activate=False):
        self.lr = lr
        self.bias = bias
        self.activate = activate

        self.layers : list[Layer]  = []
        for i in range(1, len(num_features)):
            self.layers.append(Layer(
                num_features[i-1], num_features[i],
                bias=self.bias, activate=self.activate))
        self.num_layers = len(self.layers)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        forward_result = x
        for i in range(self.num_layers):
            forward_result = self.layers[i].forward(forward_result)
        return forward_result

    def backward(self, derivative_loss):
        prev_wT_delta = derivative_loss
        for i in reversed(range(self.num_layers)):
            prev_wT_delta = self.layers[i].backward(prev_wT_delta)

    def step(self):
        for i in range(self.num_layers):
            self.layers[i].step(self.lr)


def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0] -pt[1])/1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)

    return np.array(inputs), np.array(labels).reshape(n, 1)


def generate_XOR_easy():
    inputs = []
    labels = []

    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)

        if 0.1*i == 0.5:
            continue

        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)

    return np.array(inputs), np.array(labels).reshape(21, 1)


def train(model: Network, x, y, batch_size=-1):
    print('------------------')
    print('| Start training |')
    print('------------------')
    max_epochs = 100000 # 最多訓練的次數
    loss_thres = 0.005 # 需要達到小於 0.005 的錯誤率
    bs = batch_size if batch_size!=-1 else x.shape[0] 
    train_loss = []
    epoch = 1
    while True:
        start_idx = 0
        end_idx = min(start_idx+bs, x.shape[0])
        batch_num = 0
        loss = 0
        while True:
            bx = x[start_idx:end_idx]
            by = y[start_idx:end_idx]
            pred_y = model.forward(bx)
            loss += mse_loss(pred_y, by)
            model.backward(derivative_mse_loss(pred_y, by))
            model.step()
            batch_num += 1

            start_idx = end_idx
            end_idx = min(start_idx+bs, x.shape[0])
            if start_idx >= x.shape[0]:
                break

        loss /= batch_num
        train_loss.append(loss)
        if epoch%500 == 0:
            print(f'[Epoch:{epoch:6}] [Loss: {loss:.6f}]')
        epoch += 1
        if loss<=loss_thres or epoch>=max_epochs:
            print(f'[Epoch:{epoch:6}] [Loss: {loss:.6f}]')
            break

    return model, train_loss


def test(model, x, y):
    print('-----------------')
    print('| Start testing |')
    print('-----------------')
    pred_y = model(x)
    loss = cross_entropy_loss(pred_y, y)
    pred_y_rounded = np.round(pred_y)
    pred_y_rounded[pred_y_rounded<0] = 0
    correct = np.sum(pred_y_rounded==y)
    acc = 100 * correct / len(y)
    show_result(x, y, pred_y_rounded)
    for i in range(y.shape[0]):
        print(f'Iter{i} |Ground Truth: {y[i, 0]}  |Prediction: {pred_y[i,0]}')  
    print(f'Testing loss: {loss:.3f}')
    print(f'Acc: {correct}/{len(y)} ({acc:.1f}%)')


def show_learning_curve(train_loss):
    plt.clf()
    plt.plot(np.arange(len(train_loss))+1, train_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.show()


def show_data(x, y, title):
    plt.clf()
    color = ['bo', 'ro']
    for i in range(x.shape[0]):
        plt.plot(x[i,0], x[i,1], color[int(y[i]==0)])
    plt.title(title)
    plt.axis('square')
    plt.show()


def show_result(x, y, pred_y):
    plt.subplot(1,2,1)
    plt.title('Ground truth', fontsize = 18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.subplot(1,2,2)
    plt.title('Predict result', fontsize = 18)
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.show()
    
if __name__ == '__main__':
    batch_size = -1
    lr = 1
    num_layers = '2-3-3-1'.split('-')
    num_layers = [int(n) for n in num_layers]
    bias = True
    activate = True

    os.makedirs('train_loss', exist_ok=True)
    os.makedirs('plots', exist_ok=True)

    print('Datset: linear data')
    train_x, train_y = generate_linear(100)
    show_data(train_x, train_y, title='Linear')
    test_x, test_y = generate_linear(100)
    model_linear = Network(num_layers, lr=lr, bias=bias, activate=activate)
    model_linear, train_loss = train(model_linear, train_x, train_y, batch_size)
    test(model_linear, test_x, test_y)
    show_learning_curve(train_loss)

    train_x, train_y = generate_XOR_easy()
    show_data(train_x, train_y, title='XOR')
    test_x, test_y = generate_XOR_easy()
    model_linear = Network(num_layers, lr=lr, bias=bias, activate=activate)
    model_linear, train_loss = train(model_linear, train_x, train_y, batch_size)
    test(model_linear, test_x, test_y)
    show_learning_curve(train_loss)
# %%
