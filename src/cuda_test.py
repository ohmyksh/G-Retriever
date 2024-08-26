import torch

def test_cuda_availability():
    # CUDA가 사용 가능한지 확인
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")

    if cuda_available:
        # 사용 가능한 GPU의 개수 확인
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs Available: {num_gpus}")

        # 각 GPU의 이름 확인
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {gpu_name}")

        # 현재 활성화된 디바이스 확인
        current_device = torch.cuda.current_device()
        print(f"Current CUDA Device: {current_device}")

    else:
        print("CUDA is not available. Please check your CUDA installation.")

if __name__ == "__main__":
    test_cuda_availability()