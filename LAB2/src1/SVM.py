import numpy as np
import cvxopt  # 用于求解线性规划
from process_data import load_and_process_data
from evaluation import get_micro_F1, get_macro_F1, get_acc


# 根据指定类别main_class生成1/-1标签
def svm_label(labels, main_class):
    new_label = []
    for i in range(len(labels)):
        if labels[i] == main_class:
            new_label.append(1)
        else:
            new_label.append(-1)
    return np.array(new_label)


# 实现线性回归
class SupportVectorMachine:
    '''参数初始化
    lr: 梯度更新的学习率
    Lambda: L2范数的系数
    epochs: 更新迭代的次数
    '''

    def __init__(self, kernel, C, Epsilon):
        self.kernel = kernel
        self.C = C
        self.Epsilon = Epsilon

    '''KERNEL用于计算两个样本x1,x2的核函数'''

    def KERNEL(self, x1, x2, kernel='Gauss', d=2, sigma=1):
        # d是多项式核的次数,sigma为Gauss核的参数
        K = 0
        if kernel == 'Gauss':
            K = np.exp(-(np.sum((x1 - x2) ** 2)) / (2 * sigma ** 2))
        elif kernel == 'Linear':
            K = np.dot(x1, x2)
        elif kernel == 'Poly':
            K = np.dot(x1, x2) ** d
        else:
            print('No support for this kernel')
        return K

    '''
    根据训练数据train_data,train_label（均为np数组）求解svm,并对test_data进行预测,返回预测分数，即svm使用符号函数sign之前的值
    train_data的shape=(train_num,train_dim),train_label的shape=(train_num,) train_num为训练数据的数目，train_dim为样本维度
    预测结果的数据类型应为np数组，shape=(test_num,1) test_num为测试数据的数目
    '''

    def fit(self, train_data, train_label, test_data):
        #####首先构造矩阵P#####
        p = np.ones((train_data.shape[0], train_data.shape[0]))
        for i in range(train_data.shape[0]):
            for j in range(train_data.shape[0]):
                p[i][j] = train_label[i] * train_label[j] * self.KERNEL(train_data[i], train_data[j], self.kernel)
        #####构造q,即为全为-1的列向量#####
        q = -1 * np.ones((train_data.shape[0], 1))
        #####构造h，即为C的列向量和0的列向量的拼接#####
        h = self.C * np.ones((train_data.shape[0], 1))
        h = np.r_[h, np.zeros((train_data.shape[0], 1))]
        #####构造G，要同时满足小于等于h且大于0,即为单位对角阵和负单位对角阵的拼接#####
        G = np.eye(train_data.shape[0], dtype=int)
        G = np.r_[G, -1 * G]
        ####构造A，即为y的行向量形式####
        A = train_label.reshape(1, train_data.shape[0])
        ####构造b，即为0向量####
        b = np.zeros((1, 1))
        ####使用线性规划求解器求解####
        #####先进行类型统一#####
        p = p.astype(np.double)
        q = q.astype(np.double)
        G = G.astype(np.double)
        h = h.astype(np.double)
        A = A.astype(np.double)
        b = b.astype(np.double)
        ####进行求解####
        solver = cvxopt.solvers.qp(cvxopt.matrix(p), cvxopt.matrix(q), cvxopt.matrix(G), cvxopt.matrix(h),
                                   cvxopt.matrix(A),
                                   cvxopt.matrix(b))
        alpha = np.array(solver['x'])  # 求解得到alpha

        index = np.where(alpha >= self.Epsilon)[0]  # 找到所有值不低于阈值的index
        # 利用alpha计算b
        b = np.mean(
            [train_label[i] - sum(
                [train_label[i] * alpha[i] * self.KERNEL(x, train_data[i], self.kernel) for x in train_data[index]])
             for i in index])
        ####进行预测####
        predictions = []
        for j in range(test_data.shape[0]):
            y = b + sum(
                [train_label[i] * alpha[i] * self.KERNEL(test_data[j], train_data[i], self.kernel) for i in index])
            predictions.append(y)
        y = np.array(predictions).reshape(test_data.shape[0], 1)
        return y


def main():
    # 加载训练集和测试集
    Train_data, Train_label, Test_data, Test_label = load_and_process_data()
    Train_label = [label[0] for label in Train_label]
    Test_label = [label[0] for label in Test_label]
    train_data = np.array(Train_data)
    test_data = np.array(Test_data)
    test_label = np.array(Test_label).reshape(-1, 1)
    # 类别个数
    num_class = len(set(Train_label))

    # kernel为核函数类型，可能的类型有'Linear'/'Poly'/'Gauss'
    # C为软间隔参数；
    # Epsilon为拉格朗日乘子阈值，低于此阈值时将该乘子设置为0
    kernel = 'Linear'
    C = 1
    Epsilon = 10e-5
    # 生成SVM分类器
    SVM = SupportVectorMachine(kernel, C, Epsilon)

    predictions = []
    # one-vs-all方法训练num_class个二分类器
    for k in range(1, num_class + 1):
        # 将第k类样本label置为1，其余类别置为-1
        train_label = svm_label(Train_label, k)
        # 训练模型，并得到测试集上的预测结果
        prediction = SVM.fit(train_data, train_label, test_data)
        predictions.append(prediction)
    predictions = np.array(predictions)
    # one-vs-all, 最终分类结果选择最大score对应的类别
    pred = np.argmax(predictions, axis=0) + 1

    # 计算准确率Acc及多分类的F1-score
    print("Acc: " + str(get_acc(test_label, pred)))
    print("macro-F1: " + str(get_macro_F1(test_label, pred)))
    print("micro-F1: " + str(get_micro_F1(test_label, pred)))


main()
