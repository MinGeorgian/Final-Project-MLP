# import GPUtil

# # 모든 사용 가능한 GPU 정보 가져오기
# gpus = GPUtil.getGPUs()

# for gpu in gpus:
#     print(f"GPU {gpu.id}: {gpu.name}")

# import torch

# # 현재 시스템에서 사용 가능한 CUDA 디바이스 개수 확인
# num_cuda_devices = torch.cuda.device_count()

# # 각 CUDA 디바이스의 이름과 번호 출력
# for i in range(num_cuda_devices):
#     print(f"CUDA Device {i}: {torch.cuda.get_device_name(i)}")

import torch
print(torch.__version__)
print(torch.cuda.is_available())