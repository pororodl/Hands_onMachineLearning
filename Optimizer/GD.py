import numpy as np
import matplotlib.pyplot as plt

# 定义目标函数 y = x**2
def target(x):
    return np.square(x)

def GD(x_start,lr,epoch):
    '''
    普通的梯度下降法，目的是求当x为多少时，target的值最小，具体做法是：首先计算当前x点的梯度（就是对x 求导）
    :param x_start:x的起始点
    :param lr:学习率
    :param epoch:迭代次数
    :return:
    '''
    x_loc = np.zeros(epoch+1) # 用来记录每一次迭代后x的位置

    x_loc[0] = x_start  # 记录下x 最开始的位置
    x = x_start
    for i in range(epoch):
        d_x = 2*x     # 得到梯度
        x = x-lr*d_x
        x_loc[i+1] = x
    return x_loc

def demo_0():
    # 演示怎样进行梯度下降
    line_x = np.linspace(-5, 5, 100)
    line_y = target(line_x)
    plt.figure('Gradient Descent')

    x_start = 5
    lr = 0.3
    epoch = 5

    color = 'r'

    x_loc = GD(x_start, lr, epoch)

    plt.plot(line_x, line_y, c='b')
    plt.plot(x_loc, target(x_loc), c=color, label='lr={}'.format(lr))
    plt.scatter(x_loc, target(x_loc), c=color)
    plt.legend()
    plt.show()

def demo_1():
    line_x = np.linspace(-5, 5, 100)
    line_y = target(line_x)
    plt.figure('Gradient Descent')

    x_start = 5
    lr = [0.1,0.3,0.6]
    color = ['r','g','y']
    epoch = 5

    for i in range(len(lr)):
        x_loc = GD(x_start,lr[i],epoch)
        plt.subplot(1,3,i+1)
        plt.plot(line_x, line_y, c='b')
        plt.plot(x_loc, target(x_loc), c=color[i], label='lr={}'.format(lr[i]))
        plt.scatter(x_loc, target(x_loc), c=color[i])
        plt.legend()
    plt.show()

if __name__ == '__main__':
    # demo_0()
    demo_1()