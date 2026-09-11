import os
import torch
import torch.distributed as dist

def main():
    # Initialize the distributed environment using gloo for CPU
    dist.init_process_group(backend="gloo")

    # Get dynamic rank and world size assigned by torchrun
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    print(f"[Rank {rank}/{world_size}] Process group initialized. Hostname: {os.uname().nodename}")

    # Create a tensor on CPU
    # Rank 0 gets 1.0, Rank 1 gets 2.0, etc.
    tensor = torch.tensor([float(rank + 1)], dtype=torch.float32)
    
    print(f"[Rank {rank}/{world_size}] Before all_reduce: {tensor.item()}")
    
    # Perform all_reduce (sum up the tensor across all nodes)
    # E.g. for 4 nodes, sum should be 1 + 2 + 3 + 4 = 10
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    print(f"[Rank {rank}/{world_size}] After all_reduce: {tensor.item()}")
    
    expected_sum = sum(range(1, world_size + 1))
    if tensor.item() == expected_sum:
        print(f"[Rank {rank}/{world_size}] All-reduce successful!")
    else:
        print(f"[Rank {rank}/{world_size}] All-reduce failed! Expected {expected_sum}, got {tensor.item()}")

    # Clean up
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
