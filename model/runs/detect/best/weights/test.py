import torch

# 请将此路径替换为你 best.pt 文件的实际路径
weights_path = '/usr/local/deeplearn/test/ygytest/model/runs/detect/train33/weights/best.pt' 
checkpoint = torch.load(weights_path, map_location='cpu')

# 打印出来看看
print(f"当前模型是第几轮的: epoch {checkpoint.get('epoch', 'N/A')}")
print(f"当时保存的最佳适应度分数是: {checkpoint.get('fitness', 'N/A')}")