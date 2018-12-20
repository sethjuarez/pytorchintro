import json
import time
import torch
import datetime
import numpy as np
import torch.nn as nn
from io import StringIO
import torch.nn.functional as F

from azureml.core.model import Model

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.softmax(x, dim=1)

def init():
    global model, device

    try:
        model_path = Model.get_model_path('PyTorchMNIST')
    except:
        model_path = 'outputs/model.pth'

    device = torch.device('cpu')

    model = CNN()
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.to(device)
    model.eval()

def run(raw_data):
    prev_time = time.time()
          
    post = json.loads(raw_data)

    # load and normalize image
    image = np.loadtxt(StringIO(post['image']), delimiter=',') / 255.

    # run model
    with torch.no_grad():
        x = torch.from_numpy(image).float().to(device)
        pred = model(x).detach().numpy()[0]

    # get timing
    current_time = time.time()
    inference_time = datetime.timedelta(seconds=current_time - prev_time)

    payload = {
        'time': inference_time.total_seconds(),
        'prediction': int(np.argmax(pred)),
        'scores': pred.tolist()
    }

    return payload

if __name__ == "__main__":
    img = '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,21,118,164,255,138,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,76,231,254,254,254,248,176,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,74,242,254,254,254,219,254,247,31,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,75,249,254,213,87,4,3,37,242,123,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,30,243,254,226,41,0,0,0,0,67,13,22,25,50,72,52,0,0,0,0,0,0,0,0,0,0,0,0,153,254,235,28,0,0,0,27,87,91,146,247,254,254,254,157,0,0,0,0,0,0,0,0,0,0,0,34,238,254,89,0,0,69,154,248,254,254,254,254,231,141,56,6,0,0,0,0,0,0,0,0,0,0,0,58,254,254,9,17,146,251,254,254,254,253,171,124,26,0,0,0,0,0,0,0,0,0,0,0,0,0,0,58,254,254,108,212,254,254,254,229,155,74,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,58,254,254,254,254,254,226,86,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,58,254,254,254,247,141,21,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,87,254,254,254,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,51,245,254,254,254,43,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,53,195,254,253,247,254,83,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,152,254,254,138,63,251,165,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,34,238,254,201,4,0,245,183,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,101,254,254,49,1,75,248,101,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,158,254,254,140,157,254,254,43,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,146,254,254,254,254,254,144,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,16,192,254,254,244,126,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0'
    data = {
        'image': img
    }

    init()
    out = run(json.dumps(data))
    print(out)