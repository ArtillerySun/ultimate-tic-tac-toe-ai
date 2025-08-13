import torch
from collections import OrderedDict

# Load trained model weights
torch.set_printoptions(precision=10)
state_dict = torch.load('model/uttt_model1.pth', map_location=torch.device('cpu'), weights_only=True)

with open("model/uttt_model1.txt", "w") as f:
    f.write("with torch.no_grad():\n")
    for name, param in state_dict.items():
        shape = tuple(param.shape)  # Preserve original shape
        param_list = param.tolist()  # Convert tensor to nested lists
        f.write(f"    self.{name}.copy_(torch.tensor({param_list}, dtype=torch.float32))\n")  # Write in proper format


print("Weights saved correctly in uttt_model1.txt")
