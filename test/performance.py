import torch

def CudaProfile(*function_with_args):
    times = 20
    for _ in range(times):
        for func, args in function_with_args:
            func(*args)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(times):
        for func, args in function_with_args:
            func(*args)
    end_event.record()
    # 等待事件完成
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)  # 以毫秒为单位        
    return elapsed_time/times
