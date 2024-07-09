# import torch

# @torch.library.custom_op("mylib::foo", mutates_args={"out"})
# def foo(x: torch.Tensor, out: torch.Tensor) -> None:
#     out.copy_(x.sin())


# @torch.compile(backend="inductor", fullgraph=True)
# def f(x, out): # E: Function is missing a type annotation  [no-untyped-def]
#     foo(x, out)
#     foo(out, out)
#     foo(out, out)

# x = torch.randn(3)
# out = torch.randn(3)
# f(x, out)
# assert torch.allclose(out, x.sin().sin().sin())


import torch

m = torch.library.Library("mylib", "FRAGMENT")

m.define("foo(Tensor(a!) x) -> ()")

def foo(x: torch.Tensor, out: torch.Tensor) -> None:
    out.copy_(x.sin())

m.impl("foo", foo, "CPU")


@torch.compile(backend="inductor", fullgraph=True)
def f(x, out): # E: Function is missing a type annotation  [no-untyped-def]
    foo(x, out)
    foo(out, out)
    foo(out, out)

x = torch.randn(3)
out = torch.randn(3)
f(x, out)
assert torch.allclose(out, x.sin().sin().sin())
