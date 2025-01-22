import torch
import torch.nn as nn
import torch.nn.functional as F

def check_gpu():

    print(torch)
    print(torch.__version__)
    print(torch.__file__)
    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA (GPU support) is available in this environment!")
        print("Available GPU(s):", torch.cuda.device_count())
        print("GPU Name:", torch.cuda.get_device_name(0))
    else:
        print("CUDA (GPU support) is not available.")

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 50)  # 10 input features to 50 hidden units
        self.fc2 = nn.Linear(50, 2)   # 50 hidden units to 2 output classes

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def check_gpu_by_example():

    # Check if CUDA (GPU support) is available
    
    print(torch)
    print(torch.__version__)
    print(torch.__file__)
    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA (GPU support) is available in this environment!")
        print("Available GPU(s):", torch.cuda.device_count())
        print("GPU Name:", torch.cuda.get_device_name(0))

        # Instantiate the network
        net = SimpleNN()
        print("Original device:", next(net.parameters()).device)  # Should be 'cpu'

        # Move the network to GPU
        net.to('cuda')
        print("New device:", next(net.parameters()).device)  # Should be 'cuda'

        # Perform a simple computation
        inputs = torch.randn(1, 10).to('cuda')  # Random input tensor
        outputs = net(inputs)
        print("Successfully performed a computation on GPU:", outputs)

    else:
        print("CUDA (GPU support) is not available.")

def check_mac_gpu():

    #check for gpu
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)
        print (x)
    else:
        print ("MPS device not found.")

if __name__ == '__main__':

    #check_gpu()
    check_gpu_by_example()
    check_mac_gpu()
