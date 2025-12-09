import torch
import subprocess
import sys

def check_gpu_memory():
    """检查 GPU 显存使用情况"""
    print("=" * 60)
    print("GPU 显存诊断")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("[ERROR] CUDA 不可用！")
        return
    
    num_gpus = torch.cuda.device_count()
    print(f"\n检测到 {num_gpus} 个 GPU\n")
    
    for i in range(num_gpus):
        print(f"GPU {i}:")
        print(f"  设备名称: {torch.cuda.get_device_name(i)}")
        
        # 使用 nvidia-smi 获取详细信息
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu', 
                 '--format=csv,noheader,nounits', f'--id={i}'],
                capture_output=True, text=True, check=True
            )
            parts = result.stdout.strip().split(', ')
            if len(parts) >= 4:
                mem_used = int(parts[1])
                mem_total = int(parts[2])
                gpu_util = int(parts[3])
                mem_percent = (mem_used / mem_total) * 100
                
                print(f"  显存使用: {mem_used} MB / {mem_total} MB ({mem_percent:.1f}%)")
                print(f"  GPU 利用率: {gpu_util}%")
                
                if mem_used > mem_total * 0.9:
                    print(f"  [WARNING] 显存使用率超过 90%！")
                elif mem_used > 0:
                    print(f"  [INFO] 有 {mem_used} MB 显存被占用（可能是 PyTorch 缓存）")
        except Exception as e:
            print(f"  [WARN] 无法获取详细信息: {e}")
        
        # PyTorch 显存统计
        torch.cuda.set_device(i)
        allocated = torch.cuda.memory_allocated(i) / 1024**2  # MB
        reserved = torch.cuda.memory_reserved(i) / 1024**2    # MB
        max_allocated = torch.cuda.max_memory_allocated(i) / 1024**2  # MB
        
        print(f"  PyTorch 已分配: {allocated:.2f} MB")
        print(f"  PyTorch 已保留: {reserved:.2f} MB")
        print(f"  PyTorch 峰值: {max_allocated:.2f} MB")
        print()
    
    print("=" * 60)
    print("建议操作：")
    print("1. 如果 PyTorch 已保留显存 > 0，运行清理命令：")
    print("   python -c 'import torch; torch.cuda.empty_cache(); print(\"缓存已清理\")'")
    print("2. 如果其他进程占用显存，使用以下命令查看：")
    print("   nvidia-smi")
    print("3. 如果仍有问题，尝试重启 Python 进程")
    print("=" * 60)

def clear_cache():
    """清理 PyTorch 显存缓存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print("[INFO] PyTorch 显存缓存已清理")
    else:
        print("[WARN] CUDA 不可用，无法清理缓存")

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--clear':
        clear_cache()
    else:
        check_gpu_memory()

