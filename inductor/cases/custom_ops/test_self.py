# https://github.com/pytorch/pytorch/issues/124933
import torch

@torch.library.custom_op("mylib::foo", mutates_args={"self"})
def foo(self: torch.Tensor) -> None:
    self.sin_()

x = torch.randn(3)

@torch.compile(backend="inductor", fullgraph=True)
def f(x):
    foo(x)

f(x)
