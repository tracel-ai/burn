#!/usr/bin/env -S uv run --script

# /// script
# dependencies = ["torch"]
# ///

"""Print PyTorch reference values for strided_ops_compare.rs.

Small tensors, all results printed as Rust f32/i64 slice literals.
Run: uv run --script crates/burn-flex/tests/generate_strided_reference.py
"""

import torch


def f32_arr(tensor):
    vals = tensor.detach().contiguous().float().flatten().tolist()
    return "&[" + ", ".join(f"{v:.7f}" for v in vals) + "]"


def i64_arr(tensor):
    vals = tensor.detach().contiguous().long().flatten().tolist()
    return "&[" + ", ".join(f"{v}_i64" for v in vals) + "]"


def rand_uniform(shape, seed):
    torch.manual_seed(seed)
    return torch.rand(shape) * 1.9 + 0.1


def print_input(name, t):
    print(f"#[rustfmt::skip]")
    print(f"const {name}: &[f32] = {f32_arr(t)};")


def print_suite(prefix, ta, tb):
    """Print expected values for the standard op suite."""
    print(f"    // unary")
    for name, result in [
        ("sin", torch.sin(ta)),
        ("cos", torch.cos(ta)),
        ("exp", torch.exp(ta)),
        ("log", torch.log(ta)),
        ("sqrt", torch.sqrt(ta)),
        ("abs", torch.abs(ta)),
        ("neg", -ta),
    ]:
        print(f'    cmp("{prefix}/{name}", &te.clone().{name}().into_data(), {f32_arr(result)});')

    print(f"    // binary")
    for name, op, result in [
        ("add", "+", ta + tb),
        ("mul", "*", ta * tb),
        ("sub", "-", ta - tb),
        ("div", "/", ta / tb),
    ]:
        print(f'    cmp("{prefix}/{name}", &(te.clone() {op} te2.clone()).into_data(), {f32_arr(result)});')

    print(f"    // reduce")
    for dim, result in [(2, torch.sum(ta, dim=2, keepdim=True)), (1, torch.sum(ta, dim=1, keepdim=True))]:
        print(f'    cmp("{prefix}/sum_dim{dim}", &te.clone().sum_dim({dim}).into_data(), {f32_arr(result)});')
    print(f'    cmp("{prefix}/mean_dim2", &te.clone().mean_dim(2).into_data(), {f32_arr(torch.mean(ta, dim=2, keepdim=True))});')

    argmax = torch.argmax(ta, dim=2, keepdim=True)
    print(f'    cmp_i64("{prefix}/argmax_dim2", &te.clone().argmax(2).into_data(), {i64_arr(argmax)});')

    print(f"    // softmax")
    print(f'    cmp("{prefix}/softmax_dim2", &burn_tensor::activation::softmax(te.clone(), 2).into_data(), {f32_arr(torch.softmax(ta, dim=2))});')


def main():
    # Scenario 1: step2_last
    # [2,3,4] -> step-2 on last dim -> [2,3,2] = 12 elements
    print("\n// === Scenario 1: step2_last [2,3,4] -> [2,3,2] ===")
    a = rand_uniform([2, 3, 4], seed=100)
    b = rand_uniform([2, 3, 4], seed=101)
    print_input("STEP2_LAST_A", a)
    print_input("STEP2_LAST_B", b)
    ta, tb = a[:, :, ::2], b[:, :, ::2]
    print_suite("step2_last", ta, tb)
    # matmul: [2,3,2] @ [2,2,3] -> [2,3,3]
    mm = torch.matmul(ta, ta.transpose(-1, -2))
    print(f'    cmp("step2_last/matmul", &te.clone().matmul(te.transpose()).into_data(), {f32_arr(mm)});')

    # Scenario 2: step3_mid
    # [2,6,3] -> step-3 on dim 1 -> [2,2,3] = 12 elements
    print("\n// === Scenario 2: step3_mid [2,6,3] -> [2,2,3] ===")
    a = rand_uniform([2, 6, 3], seed=200)
    b = rand_uniform([2, 6, 3], seed=201)
    print_input("STEP3_MID_A", a)
    print_input("STEP3_MID_B", b)
    ta, tb = a[:, ::3, :], b[:, ::3, :]
    print_suite("step3_mid", ta, tb)

    # Scenario 3: narrow_transpose
    # [2,4,3] -> narrow(1,1,2) -> [2,2,3] -> swap(1,2) -> [2,3,2] = 12 elements
    print("\n// === Scenario 3: narrow_transpose [2,4,3] -> [2,3,2] ===")
    a = rand_uniform([2, 4, 3], seed=300)
    b = rand_uniform([2, 4, 3], seed=301)
    print_input("NARROW_TRANS_A", a)
    print_input("NARROW_TRANS_B", b)
    ta = a.narrow(1, 1, 2).transpose(1, 2)
    tb = b.narrow(1, 1, 2).transpose(1, 2)
    print_suite("narrow_trans", ta, tb)

    # Scenario 4: step2_transpose
    # [2,3,4] -> step-2 last -> [2,3,2] -> swap(1,2) -> [2,2,3] = 12 elements
    print("\n// === Scenario 4: step2_transpose [2,3,4] -> [2,2,3] ===")
    a = rand_uniform([2, 3, 4], seed=400)
    b = rand_uniform([2, 3, 4], seed=401)
    print_input("STEP2_TRANS_A", a)
    print_input("STEP2_TRANS_B", b)
    ta = a[:, :, ::2].transpose(1, 2)
    tb = b[:, :, ::2].transpose(1, 2)
    print_suite("step2_trans", ta, tb)

    # Scenario 5: expand_s0
    # [1,3,2] -> expand(2,3,2) -> [2,3,2] = 12 elements
    print("\n// === Scenario 5: expand_s0 [1,3,2] -> [2,3,2] ===")
    a = rand_uniform([1, 3, 2], seed=500)
    b = rand_uniform([1, 3, 2], seed=501)
    print_input("EXPAND_S0_A", a)
    print_input("EXPAND_S0_B", b)
    ta, tb = a.expand(2, 3, 2), b.expand(2, 3, 2)
    print(f"    // unary subset")
    print(f'    cmp("expand_s0/sin", &te.clone().sin().into_data(), {f32_arr(torch.sin(ta))});')
    print(f'    cmp("expand_s0/exp", &te.clone().exp().into_data(), {f32_arr(torch.exp(ta))});')
    print(f'    cmp("expand_s0/neg", &te.clone().neg().into_data(), {f32_arr(-ta)});')
    print(f'    cmp("expand_s0/abs", &te.clone().abs().into_data(), {f32_arr(torch.abs(ta))});')
    print(f"    // binary")
    print(f'    cmp("expand_s0/add", &(te.clone() + te2.clone()).into_data(), {f32_arr(ta + tb)});')
    print(f'    cmp("expand_s0/mul", &(te.clone() * te2.clone()).into_data(), {f32_arr(ta * tb)});')
    print(f"    // reduce")
    print(f'    cmp("expand_s0/sum_dim0", &te.clone().sum_dim(0).into_data(), {f32_arr(torch.sum(ta, dim=0, keepdim=True))});')
    print(f'    cmp("expand_s0/sum_dim2", &te.clone().sum_dim(2).into_data(), {f32_arr(torch.sum(ta, dim=2, keepdim=True))});')
    print(f'    cmp("expand_s0/mean_dim1", &te.clone().mean_dim(1).into_data(), {f32_arr(torch.mean(ta, dim=1, keepdim=True))});')
    print(f"    // softmax")
    print(f'    cmp("expand_s0/softmax_dim2", &burn_tensor::activation::softmax(te.clone(), 2).into_data(), {f32_arr(torch.softmax(ta, dim=2))});')

    # Scenario 6: gather on step-sliced
    # [2,2,4] -> step-2 last -> [2,2,2], gather dim=2 with [2,2,1] indices
    print("\n// === Scenario 6: gather on step-sliced [2,2,4] -> [2,2,2] ===")
    a = rand_uniform([2, 2, 4], seed=600)
    print_input("GATHER_INPUT", a)
    ta = a[:, :, ::2]
    indices = torch.tensor([[[1], [0]], [[0], [1]]])
    result = torch.gather(ta, 2, indices)
    print(f'    cmp("gather/step2", &te.gather(2, idx).into_data(), {f32_arr(result)});')


if __name__ == "__main__":
    main()
