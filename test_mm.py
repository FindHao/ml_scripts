
def work():
    # A simple mm test 
    import torch
    n=4096
    x = torch.ones((n, n), dtype=torch.float32, device="cuda")
    y = torch.ones((n, n),dtype=torch.float32, device="cuda")
    # warmup
    for i in range(10):
        torch.mm(x, y)
    for i in range(2):
        torch.mm(x, y)


if __name__ == "__main__":
    work()