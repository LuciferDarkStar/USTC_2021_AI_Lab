from process_data import load_and_process_data
from evaluation import get_macro_F1, get_micro_F1, get_acc
import numpy as np


# 实现线性回归的类
class LinearClassification:
    '''参数初始化
    lr: 梯度更新的学习率
    Lambda: L2范数的系数
    epochs: 更新迭代的次数
    '''

    def __init__(self, lr=0.05, Lambda=0.001, epochs=1000):
        self.lr = lr
        self.Lambda = Lambda
        self.epochs = epochs

    '''根据训练数据train_features,train_labels计算梯度更新参数W'''

    def fit(self, train_features, train_labels):
        x = np.c_[np.ones(train_features.shape[0]), train_features]  # 构建x矩阵
        w = np.random.random(train_features.shape[1] + 1)  # 随机生成w，其值在（0，1）
        # print(w)
        w = w.reshape(-1, 1)  # 转换为列向量

        now_epochs = self.epochs  # 取训练轮数
        while now_epochs > 0:
            xw = x.dot(w)  # 计算x*w
            xw_y = xw - train_labels  # 计算xw-y
            xw_y_T = xw_y.reshape(xw_y.shape[1], xw_y.shape[0])  # 转置
            gradient = 2*np.dot(xw_y_T, x)/len(train_features) + 2*self.Lambda * w.reshape(1, -1)  # 得到梯度
            w -= self.lr * gradient.reshape(-1, 1)
            now_epochs -= 1

        self.w = w  # 训练结束得到w

    '''根据训练好的参数对测试数据test_features进行预测，返回预测结果
    预测结果的数据类型应为np数组，shape=(test_num,1) test_num为测试数据的数目'''

    def predict(self, test_features):
        x = np.c_[np.ones(test_features.shape[0]), test_features]  # 构造x
        i = test_features.shape[0]  # 测试数据的数量
        Prediction = []  # 预测结果
        j = 0
        # 预测类别
        while j < i:
            y = x[j].dot(self.w)
            if y >= 2.5:
                Prediction.append(3)
            elif y >= 1.75:
                Prediction.append(2)
            else:
                Prediction.append(1)
            j += 1
        Prediction = np.array(Prediction).reshape(i, 1)  # 格式化
        return Prediction


def main():
    # 加载训练集和测试集
    train_data, train_label, test_data, test_label = load_and_process_data()
    lR = LinearClassification()
    lR.fit(train_data, train_label)  # 训练模型
    pred = lR.predict(test_data)  # 得到测试集上的预测结果

    # 计算准确率Acc及多分类的F1-score
    print("Acc: " + str(get_acc(test_label, pred)))
    print("macro-F1: " + str(get_macro_F1(test_label, pred)))
    print("micro-F1: " + str(get_micro_F1(test_label, pred)))


main()
