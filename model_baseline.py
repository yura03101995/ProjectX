import torch.nn as nn

class DNN(nn.Module):
    '''
    Dot Neural Net
    '''
    def __init__(self, seq_sizes_dot):
        '''
        Init model.
        Arguments:
            seq_sizes_dot - tuple, sequence shape of fully-connections layers
        '''
        super(DNN, self).__init__()
        self.model = None
        for inputs, outputs in zip(seq_sizes_dot, seq_sizes_dot[1:-1]):
            if self.model is not None:
                self.model = nn.Sequential(*self.model, 
                                    nn.Linear(inputs, outputs, bias=True),
                                    nn.Sigmoid())
            else:
                self.model = nn.Sequential(nn.Linear(inputs, outputs, bias=True), nn.Sigmoid())
        self.model = nn.Sequential(*self.model, 
                                nn.Linear(seq_sizes_dot[-2], seq_sizes_dot[-1]), 
                                nn.Softmax())

    def forward(self, x):
        x = self.model(x)
        return x

class CNN(nn.Module):
    '''
    Conv Neural Net
    '''
    def __init__(self, num_classes=3):
        '''
        Init model.
        Arguments:
            seq_sizes_dot - tuple, sequence shape of fully-connections layers
        '''
        super(CNN, self).__init__()
        
        self.conv2d_1 = nn.Conv2d(1, 3, kernel_size=(10, 39), stride=4, bias=True)
        self.relu_1 = nn.ReLU()
        self.conv2d_2 = nn.Conv2d(3, 15, kernel_size=(10, 1), stride=4, bias=True)
        self.relu_2 = nn.ReLU()
        self.conv2d_3 = nn.Conv2d(15, 65, kernel_size=(10, 1), stride=4, bias=True)
        #self.avgpool = nn.AvgPool()
        self.fc = nn.Linear(65, num_classes, bias=True)
        self.softmax = nn.Softmax()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode='fan_out')#, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)

    def forward(self, x):
        x = self.conv2d_1(x)
        x = self.relu_1(x)
        x = self.conv2d_2(x)
        x = self.relu_2(x)
        x = self.conv2d_3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.softmax(x)
        return x