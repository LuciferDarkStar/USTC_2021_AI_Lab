import numpy as np
import math
from collections import Counter
from process_data import load_and_process_data
from evaluation import get_micro_F1, get_macro_F1, get_acc


class NaiveBayes:
    '''参数初始化
    Pc: P(c) 每个类别c的概率分布
    Pxc: P(c|x) 每个特征的条件概率
    '''

    def __init__(self):
        self.Pc = {}
        self.Pxc = {}

    '''
    通过训练集计算先验概率分布p(c)和条件概率分布p(x|c)
    建议全部取log，避免相乘为0
    '''

    def fit(self, traindata, trainlabel, featuretype):
        #################预处理数据#################
        D_c = {}  # 统计各类的总数
        for i in range(1, 4):  # 初始化为0
            D_c[i] = 0
        D_c_sex = {}  # 统计各类不同性别的数量
        for i in range(1, 4):  # 初始化为0
            for j in range(1, 4):
                D_c_sex[i, j] = 0
        column = {}  # 不同类的连续型属性的值集合
        for i in range(traindata.shape[0]):  # 遍历数据，统计各种数量
            D_c[int(trainlabel[i])] += 1  # 各类的总数
            D_c_sex[(int(trainlabel[i]), int(traindata[i][0]))] += 1  # 各类不同性别的总数

            for j in range(1, 8):  # 统计连续型属性
                if (int(trainlabel[i]), j) not in column.keys():
                    column[int(trainlabel[i]), j] = np.array(float(traindata[i][j]))  # 第一次遇到该类属性
                else:
                    column[int(trainlabel[i]), j] = np.append(column[int(trainlabel[i]), j],
                                                              float(traindata[i][j]))  # 后续遇到进行加入即可
        Sum_D = D_c[1] + D_c[2] + D_c[3]  # 总数
        #################预处理完成#################

        #################计算概率#################
        for i in range(1, 4):  # 计算先验概率
            self.Pc[i] = math.log((D_c[i] + 1) / (Sum_D + 3))
            for j in range(0, 8):  # 计算条件概率
                if j == 0:  # 性别
                    for m in range(1, 4):
                        self.Pxc[i, j, m] = math.log((D_c_sex[i, m] + 1) / (D_c[i] + 3))
                else:  # 连续型属性，这里我使用高斯分布来表示
                    avg = np.average(column[i, j])  # 计算平均值
                    var = np.var(column[i, j])  # 计算方差
                    self.Pxc[i, j] = (avg, var)
        #################计算完毕#################

    '''
    根据先验概率分布p(c)和条件概率分布p(x|c)对新样本进行预测
    返回预测结果,预测结果的数据类型应为np数组，shape=(test_num,1) test_num为测试数据的数目
    feature_type为0-1数组，表示特征的数据类型，0表示离散型，1表示连续型
    '''

    def predict(self, features, featuretype):
        y = []
        for i in range(features.shape[0]):
            argmax = 0  # 初始化目标最大值
            c = 0  # 初始化类别
            for j in range(1, 4):
                h = self.Pc[j] + self.Pxc[j, 0, int(features[i][0])]  # 因为都转换为log形式了，所以这里使用加法，先验概率加性别概率
                for m in range(1, 8):
                    (avg, var) = self.Pxc[j, m]
                    std = np.sqrt(var)
                    #####计算高斯分布概率######
                    t = 1 / (((2 * math.pi) ** 0.5) * std)
                    e = math.exp(-0.5 * ((features[i][m] - avg) ** 2) / var)
                    h += math.log(t * e)
                    #####计算结束######
                if h > argmax:  # 比较预测结果
                    argmax = h
                    c = j
            y.append(c)
        y = np.array(y).reshape(features.shape[0], 1)
        return y


def main():
    # 加载训练集和测试集
    train_data, train_label, test_data, test_label = load_and_process_data()
    feature_type = [0, 1, 1, 1, 1, 1, 1, 1]  # 表示特征的数据类型，0表示离散型，1表示连续型

    Nayes = NaiveBayes()
    Nayes.fit(train_data, train_label, feature_type)  # 在训练集上计算先验概率和条件概率

    pred = Nayes.predict(test_data, feature_type)  # 得到测试集上的预测结果

    # 计算准确率Acc及多分类的F1-score
    print("Acc: " + str(get_acc(test_label, pred)))
    print("macro-F1: " + str(get_macro_F1(test_label, pred)))
    print("micro-F1: " + str(get_micro_F1(test_label, pred)))


main()
