import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import MNIST

# 禁止import除了torch以外的其他包，依赖这几个包已经可以完成实验了

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Mixer_Layer(nn.Module):
    def __init__(self, patch_size, hidden_dim):
        super(Mixer_Layer, self).__init__()
        ########################################################################
        # 这里需要写Mixer_Layer（layernorm，mlp1，mlp2，skip_connection）
        self.patch_size = (28 // patch_size) ** 2
        self.hidden_dim = hidden_dim
        self.layernorm = nn.LayerNorm(self.hidden_dim)
        # 行列交替两种类型的MLP
        self.fn1 = nn.Sequential(
            nn.Linear(self.patch_size, self.hidden_dim * 3),
            nn.GELU(),
            nn.Dropout(0),
            nn.Linear(self.hidden_dim * 3, self.patch_size),
            nn.Dropout(0)
        )
        self.fn2 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 3),
            nn.GELU(),
            nn.Dropout(0),
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.Dropout(0)
        )
        ########################################################################

    def forward(self, x):
        ########################################################################
        temp1 = torch.transpose(self.layernorm(x), 1, 2)
        temp1 = self.fn1(temp1)
        temp1 = torch.transpose(temp1, 1, 2) + x
        temp2 = self.fn2(self.layernorm(temp1))
        return temp2 + temp1
        ########################################################################


class MLPMixer(nn.Module):
    def __init__(self, patch_size, hidden_dim, depth):
        super(MLPMixer, self).__init__()
        assert 28 % patch_size == 0, 'image_size must be divisible by patch_size'
        assert depth > 1, 'depth must be larger than 1'
        ########################################################################
        # 这里写Pre-patch Fully-connected, Global average pooling, fully connected
        # 对图片进行拆分
        self.folding = nn.Conv2d(kernel_size=patch_size, stride=patch_size, in_channels=1,
                                 out_channels=hidden_dim)
        self.mixer_layer = nn.ModuleList(
            [Mixer_Layer(patch_size=patch_size, hidden_dim=hidden_dim) for i in range(depth)])
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.Classifier = nn.Linear(hidden_dim, 10)
        ########################################################################

    def forward(self, data):
        ########################################################################
        # 注意维度的变化
        temp = self.folding(data)
        temp = torch.transpose(temp.view(temp.shape[0], temp.shape[1], temp.shape[2] * temp.shape[3]), 1, 2)
        for f in self.mixer_layer:
            temp = f(temp)
        return self.Classifier(self.layernorm(temp).mean(dim=1))
        ########################################################################


def train(model, train_loader, optimizer, n_epochs, criterion):
    model.train()
    for epoch in range(n_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            ########################################################################
            # 计算loss并进行优化
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
            ########################################################################
            if batch_idx % 100 == 0:
                print('Train Epoch: {}/{} [{}/{}]\tLoss: {:.6f}'.format(
                    epoch, n_epochs, batch_idx * len(data), len(train_loader.dataset), loss.item()))


def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0.
    num_correct = 0  # correct的个数
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            ########################################################################
            # 需要计算测试集的loss和accuracy
            for i in range(model(data).shape[0]):
                if torch.max(model(data), 1)[1][i] == target[i]:
                    num_correct = num_correct + 1
            test_loss = test_loss + criterion(model(data), target)
        accuracy = num_correct / len(test_loader.dataset)
        test_loss = test_loss / len(test_loader.dataset)
        ########################################################################
        print("Test set: Average loss: {:.4f}\t Acc {:.2f}".format(test_loss.item(), accuracy))


if __name__ == '__main__':
    n_epochs = 5
    batch_size = 128
    learning_rate = 1e-3

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])

    trainset = MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2,
                                               pin_memory=True)

    testset = MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2,
                                              pin_memory=True)

    ########################################################################
    model = MLPMixer(patch_size=7, hidden_dim=14, depth=3).to(device)  # 参数自己设定，其中depth必须大于1
    # 这里需要调用optimizer，criterion(交叉熵)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    ########################################################################

    train(model, train_loader, optimizer, n_epochs, criterion)
    test(model, test_loader, criterion)
