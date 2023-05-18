from torch import nn, randn
from torchsummary import summary
import torch



class CNN(nn.Module):

    def __init__(self, task='classification', num_classes=8):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            #nn.BatchNorm3d(64),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            #nn.BatchNorm3d(64),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            #nn.BatchNorm3d(64),
            nn.MaxPool1d(kernel_size=2),
            
        )
        
        self.dense = nn.Sequential(
            nn.LazyLinear(2000),
            nn.Linear(2000, 1000),
            nn.Linear(1000, 500),
        )

        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax(dim=1)
        self.regression_head = nn.LazyLinear(1)
        self.classification_head = nn.LazyLinear(num_classes)
        
        self.out = self.regression_head if task != 'classification' else self.classification_head

    def forward(self, input_data):
        x = self.conv(input_data)
        
        x = self.dense(x)
        x = self.flatten(x)
        x = self.out(x)
        

        return x

if __name__ == "__main__":
    # print torch version
    
    model = CNN(task = 'regression').cuda()
    print(model)
    summary(model, (1, 1000))
    
    pseudo_input = randn(1, 1, 1000).cuda()
    
    output = model(pseudo_input)
    print(output.shape, output)
    