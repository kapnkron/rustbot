import torch
import torch.nn as nn
import torch.nn.functional as F # Often used, good to have

class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim1=64, hidden_dim2=32, output_dim=3, dropout_rate=0.2):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
            
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
            
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
            
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
            
        x = self.fc3(x)
        x = self.log_softmax(x)
        return x

# Example of how you might instantiate it (not needed here, just for context):
# if __name__ == '__main__':
#     # Example:
#     # scaler_X = joblib.load('scaler_X_trading.pkl')
#     # input_features = scaler_X.n_features_in_
#     # model = Net(input_dim=input_features)
#     # print(model)
#     pass
