import torch
import math
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class MLP:
    # 初始化
    def __init__(self, x, y, w1, w2, w3, lr=0.01, epochs=500):
        self.x = x
        self.y = y
        self.lr = lr
        self.epochs = epochs
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

    # 前向传播
    def Forwarding(self):
        ####5-4####
        wx = torch.mm(self.w1, torch.transpose(self.x, 0, 1))  # 计算wx12
        self.y2 = torch.transpose(torch.div(1, 1 + torch.exp(-wx)), 0, 1)  # 第二层的激活函数输出

        ####4-4####
        wx = torch.mm(self.w2, torch.transpose(self.y2, 0, 1))  # 计算wx23
        self.y3 = torch.transpose(torch.div(1, 1 + torch.exp(-wx)), 0, 1)  # 第三层的激活函数输出

        ####4-3####
        wx = torch.mm(self.w3, torch.transpose(self.y3, 0, 1))  # 计算wx34
        y4 = torch.exp(torch.transpose(wx, 0, 1))
        s = y4.sum(1)  # 求分母上的和
        self.y4 = torch.div(y4, s.reshape(-1, 1))  # 输出结果
        # print(self.y4)

        ####求loss####
        self.loss = torch.zeros(1)  # 初始化
        for j in range(self.x.shape[0]):
            self.loss = self.loss - torch.log(self.y4[j][self.y[j]])
        self.loss = self.loss / self.x.shape[0]
        # print(self.loss)

    # 反向传播
    def Backwarding(self):
        self.WL3 = self.y4
        for j in range(self.y4.shape[0]):
            self.WL3[j][self.y[j]] = self.WL3[j][self.y[j]] - 1
        self.WL2 = torch.mm(self.WL3, self.w3)
        self.WL3 = torch.mm(torch.transpose(self.WL3, 0, 1), self.y3)
        self.WL1 = torch.mm(self.WL2 * (self.y3 * (1 - self.y3)), self.w2)
        self.WL2 = torch.mm(torch.transpose(self.WL2 * (self.y3 * (1 - self.y3)), 0, 1), self.y2)
        self.WL1 = torch.mm(torch.transpose(self.WL1 * (self.y2 * (1 - self.y2)), 0, 1), self.x)

    # 梯度下降法
    def Gradient_des(self):
        loss = []
        # 梯度对比，这里只对比一轮
        for i in range(self.epochs):
            self.Forwarding()
            temp = 0
            for j in range(self.x.shape[0]):
                temp = temp - math.log(self.y4[j][self.y[j]])
            temp = temp / self.x.shape[0]
            loss.append(temp)
            self.Backwarding()
            if i == 0:
                self.Compare()
            # 梯度下降
            self.w1 = self.w1 - self.lr * (self.WL1 / self.x.shape[0])
            self.w2 = self.w2 - self.lr * (self.WL2 / self.x.shape[0])
            self.w3 = self.w3 - self.lr * (self.WL3 / self.x.shape[0])
        plt.plot(loss)
        plt.show()

    # 比对梯度
    def Compare(self):
        self.loss.backward()
        print("自动计算W3：")
        print(self.w3.grad.transpose(0, 1))
        print("手动计算W3：")
        print((self.WL3 / self.x.shape[0]).transpose(0, 1))
        print("自动计算W2：")
        print(self.w2.grad.transpose(0, 1))
        print("手动计算W2：")
        print((self.WL2 / self.x.shape[0]).transpose(0, 1))
        print("自动计算W1：")
        print(self.w1.grad.transpose(0, 1))
        print("手动计算W1：")
        print((self.WL1 / self.x.shape[0]).transpose(0, 1))


def main():
    # 输入数据，随机生成
    x = torch.rand(size=(100, 5), requires_grad=True)
    y = torch.randint(3, size=(100, 1))
    w1 = torch.rand(size=(4, 5), requires_grad=True)
    w2 = torch.rand(size=(4, 4), requires_grad=True)
    w3 = torch.rand(size=(3, 4), requires_grad=True)
    m = MLP(x, y, w1, w2, w3)
    m.Gradient_des()


main()
