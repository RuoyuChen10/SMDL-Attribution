import torch, platform
print("torch:", torch.__version__)
print("cuda in torch wheels:", torch.version.cuda)
print("cuda available?", torch.cuda.is_available(), "num_gpus:", torch.cuda.device_count())
print("cudnn:", torch.backends.cudnn.version())
print("python:", platform.python_version())