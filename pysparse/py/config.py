import torch

def info(*args):
    print("\x1b[90m[PYINFO]", *args, "\x1b[0m")

if torch.cuda.is_available():
    DEV = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    info("Device set to " + str(torch.cuda.get_device_name(DEV)))
else:
    DEV = torch.device('cpu')
    info("Device set to CPU")