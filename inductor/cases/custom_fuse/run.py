from typing import List, Callable, Optional, Tuple

import torch
from torch._inductor.pattern_matcher import PatternMatcherPass, register_replacement, fwd_only
from torch._higher_order_ops.auto_functionalize import auto_functionalized

torch.set_default_device("cuda")


@torch.library.custom_op("vllm::fused_rms_norm_quant_static", mutates_args=['result', 'scale'])
def fused_rms_norm_quant_static(result: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor,
                                azp: torch.Tensor, epsilon: float) -> None:
    print("vllm::fused_rms_norm_quant_static")
    # bogus implementation doesn't matter
    result_rms = torch.mul(input, weight) + epsilon
    result = torch.mul(result_rms, scale).to(torch.int8)
    scale.fill_(0.5)


@torch.library.custom_op("vllm::rms_norm", mutates_args=['result'])
def rms_norm(result: torch.Tensor, input: torch.Tensor, weight: torch.Tensor,
              epsilon: float) -> None:
    # bogus implementation doesn't matter
    result = torch.mul(input, weight) + epsilon


@torch.library.custom_op("vllm::static_scaled_int8_quant", mutates_args=['result', 'scale'])
def static_scaled_int8_quant(result: torch.Tensor,
                              input: torch.Tensor,
                              scale: torch.Tensor,
                              azp: Optional[torch.Tensor] = None) -> None:
    # bogus implementation doesn't matter
    result = torch.mul(input, scale).to(torch.int8)
    scale.fill_(0.5)


def rms_pattern_static(result: torch.Tensor, result_rms: torch.Tensor, input: torch.Tensor, weight: torch.Tensor,
                       scale: torch.Tensor):
    at1 = auto_functionalized(torch.ops.vllm.rms_norm.default, result=result_rms, input=input, weight=weight, epsilon=1e-6)
    at2 = auto_functionalized(torch.ops.vllm.static_scaled_int8_quant.default, result=result, input=at1[1], scale=scale,
                              azp=None)

    return at2[1], at2[2]


def rms_replacement_static(result: torch.Tensor, result_rms: torch.Tensor, input: torch.Tensor, weight: torch.Tensor,
                           scale: torch.Tensor):
    at = auto_functionalized(torch.ops.vllm.fused_rms_norm_quant_static.default, result=result, input=input, weight=weight,
                             epsilon=1e-6, scale=scale, azp=None)

    return at[1], at[2]


def empty_bf16(*args, **kwargs):
    return torch.empty(*args, **kwargs, dtype=torch.bfloat16)


def empty_int8(*args, **kwargs):
    return torch.empty(*args, **kwargs, dtype=torch.int8)


my_patterns = PatternMatcherPass()
inputs = [empty_int8(5, 4), empty_bf16(5, 4), empty_bf16(5, 4), empty_bf16(5, 1), torch.empty(1, 1)]
register_replacement(rms_pattern_static, rms_replacement_static, inputs, fwd_only, my_patterns)

def custom_pass(graph: torch.fx.Graph) -> torch.fx.Graph:
    count = my_patterns.apply(graph)
    print(f"Count: {count}")
    graph.eliminate_dead_code()
    graph.print_tabular()
    return graph

def custom_backend(graph: torch.fx.GraphModule, example_inputs: List[torch.Tensor]) -> Callable:
    from torch._inductor import config
    current_config = config.shallow_copy_dict()
    from torch._inductor.compile_fx import compile_fx
    current_config['post_grad_custom_post_pass'] = custom_pass
    return compile_fx(graph, example_inputs, config_patches=current_config)

@torch.compile(backend=custom_backend)
def my_func_static(x, w, epsilon):
    # 创建所需的输出张量
    quant_result = torch.empty_like(x, dtype=torch.int8)
    result_rms = torch.empty_like(x, dtype=torch.bfloat16)  # 新增：RMS norm的中间结果
    scale = torch.ones((1, 1))
    
    # 确保输入tensor的dtype正确
    x = x.to(torch.bfloat16)
    w = w.to(torch.bfloat16)
    
    # 直接调用rms_pattern_static
    quant_result, scale = rms_pattern_static(
        result=quant_result,
        result_rms=result_rms,
        input=x,
        weight=w,
        scale=scale
    )
    
    return quant_result, scale


print("Run my_func_static")
inputs = [torch.empty((5, 4)), torch.empty((5, 1)), 1e-6]
print(my_func_static(*inputs))
