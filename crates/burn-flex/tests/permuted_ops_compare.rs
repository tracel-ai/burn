//! Diagnostic test: compare Flex vs NdArray on permuted (non-contiguous) tensors.
//!
//! RF-DETR uses heavy permute (113x) to create non-contiguous 4D tensors for multi-head
//! attention, then runs matmul/softmax/layernorm/binary ops on them. This test isolates
//! each op with both contiguous and permuted inputs.
//!
//! Run with: cargo test --test permuted_ops_compare -- --nocapture

use burn_tensor::backend::Backend;
use burn_tensor::{Distribution, Tensor, TensorData};

type E = burn_flex::Flex;
type N = burn_ndarray::NdArray;

const TOL: f32 = 1e-5;

fn compare(name: &str, flex_data: &TensorData, ndarray_data: &TensorData) -> (f32, f32) {
    let e_vals: Vec<f32> = flex_data.to_vec().unwrap();
    let n_vals: Vec<f32> = ndarray_data.to_vec().unwrap();

    assert_eq!(
        e_vals.len(),
        n_vals.len(),
        "{}: length mismatch (flex={}, ndarray={})",
        name,
        e_vals.len(),
        n_vals.len()
    );

    let mut max_diff: f32 = 0.0;
    let mut sum_diff: f32 = 0.0;
    for (i, (&e, &n)) in e_vals.iter().zip(n_vals.iter()).enumerate() {
        let diff = (e - n).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        sum_diff += diff;
        if diff > TOL * 10.0 {
            // Print first few large diffs for debugging
            if sum_diff < 100.0 * TOL {
                eprintln!("  [{name}] idx={i}: flex={e:.8}, ndarray={n:.8}, diff={diff:.8}");
            }
        }
    }
    let mean_diff = if e_vals.is_empty() {
        0.0
    } else {
        sum_diff / e_vals.len() as f32
    };

    let status = if max_diff > TOL { "FAIL" } else { "ok" };
    eprintln!("  {status:>4} {name}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}");

    assert!(
        max_diff <= TOL,
        "{name}: max_diff {max_diff:.2e} exceeds tolerance {TOL:.2e}"
    );

    (max_diff, mean_diff)
}

/// Create identical random TensorData for seeded reproducibility.
fn rand_data(shape: &[usize], seed: u64) -> TensorData {
    let dev: <N as Backend>::Device = Default::default();
    N::seed(&dev, seed);
    let t: Tensor<N, 4> = Tensor::random(shape.to_vec(), Distribution::Uniform(-1.0, 1.0), &dev);
    t.into_data()
}

// ============================================================================
// Test 1: Matmul on permuted 4D tensors (attention Q*K^T pattern)
// ============================================================================

#[test]
fn test_matmul_4d_permuted() {
    eprintln!("\n=== Matmul 4D Permuted (attention Q*K^T pattern) ===");

    // Shape: [batch=2, heads=4, seq=8, head_dim=16]
    // After permute [0,2,1,3]: [2, 8, 4, 16]
    // Matmul: Q [2,4,8,16] @ K^T [2,4,16,8] -> [2,4,8,8]
    let b = 2;
    let h = 4;
    let s = 8;
    let d = 16;

    // Create data in [B, S, H, D] shape, then permute to [B, H, S, D]
    let q_data = rand_data(&[b, s, h, d], 42);
    let k_data = rand_data(&[b, s, h, d], 43);

    // Test contiguous first
    {
        let q_e: Tensor<E, 4> = Tensor::from_data(q_data.clone(), &Default::default());
        let q_n: Tensor<N, 4> = Tensor::from_data(q_data.clone(), &Default::default());
        let k_e: Tensor<E, 4> = Tensor::from_data(k_data.clone(), &Default::default());
        let k_n: Tensor<N, 4> = Tensor::from_data(k_data.clone(), &Default::default());

        let result_e = q_e.matmul(k_e.transpose());
        let result_n = q_n.matmul(k_n.transpose());

        compare(
            "matmul_4d_contiguous",
            &result_e.into_data(),
            &result_n.into_data(),
        );
    }

    // Test permuted (the RF-DETR pattern)
    {
        let q_e: Tensor<E, 4> = Tensor::from_data(q_data.clone(), &Default::default());
        let q_n: Tensor<N, 4> = Tensor::from_data(q_data.clone(), &Default::default());
        let k_e: Tensor<E, 4> = Tensor::from_data(k_data.clone(), &Default::default());
        let k_n: Tensor<N, 4> = Tensor::from_data(k_data.clone(), &Default::default());

        // Permute [B,S,H,D] -> [B,H,S,D]
        let q_e = q_e.permute([0, 2, 1, 3]);
        let q_n = q_n.permute([0, 2, 1, 3]);
        let k_e = k_e.permute([0, 2, 1, 3]);
        let k_n = k_n.permute([0, 2, 1, 3]);

        // Q @ K^T: [B,H,S,D] @ [B,H,D,S] -> [B,H,S,S]
        let result_e = q_e.matmul(k_e.transpose());
        let result_n = q_n.matmul(k_n.transpose());

        compare(
            "matmul_4d_permuted",
            &result_e.into_data(),
            &result_n.into_data(),
        );
    }
}

// ============================================================================
// Test 2: Softmax on permuted 4D tensor (dim 3)
// ============================================================================

#[test]
fn test_softmax_4d_permuted() {
    eprintln!("\n=== Softmax 4D Permuted ===");

    let data = rand_data(&[2, 8, 4, 16], 44);

    // Contiguous
    {
        let t_e: Tensor<E, 4> = Tensor::from_data(data.clone(), &Default::default());
        let t_n: Tensor<N, 4> = Tensor::from_data(data.clone(), &Default::default());

        let result_e = burn_tensor::activation::softmax(t_e, 3);
        let result_n = burn_tensor::activation::softmax(t_n, 3);

        compare(
            "softmax_contiguous_dim3",
            &result_e.into_data(),
            &result_n.into_data(),
        );
    }

    // Permuted: [2,8,4,16] -> [2,4,8,16]
    {
        let t_e: Tensor<E, 4> = Tensor::from_data(data.clone(), &Default::default());
        let t_n: Tensor<N, 4> = Tensor::from_data(data.clone(), &Default::default());

        let t_e = t_e.permute([0, 2, 1, 3]);
        let t_n = t_n.permute([0, 2, 1, 3]);

        let result_e = burn_tensor::activation::softmax(t_e, 3);
        let result_n = burn_tensor::activation::softmax(t_n, 3);

        compare(
            "softmax_permuted_dim3",
            &result_e.into_data(),
            &result_n.into_data(),
        );
    }

    // Permuted with softmax on dim 2 (attention weights)
    {
        let data = rand_data(&[2, 4, 8, 8], 45);
        let t_e: Tensor<E, 4> = Tensor::from_data(data.clone(), &Default::default());
        let t_n: Tensor<N, 4> = Tensor::from_data(data.clone(), &Default::default());

        // Softmax on last dim of attention scores
        let result_e = burn_tensor::activation::softmax(t_e, 3);
        let result_n = burn_tensor::activation::softmax(t_n, 3);

        compare(
            "softmax_attention_scores",
            &result_e.into_data(),
            &result_n.into_data(),
        );
    }
}

// ============================================================================
// Test 3: sum_dim / mean_dim on permuted 4D tensor
// ============================================================================

#[test]
fn test_reduce_4d_permuted() {
    eprintln!("\n=== Reduce (sum_dim/mean_dim) 4D Permuted ===");

    let data = rand_data(&[2, 8, 4, 16], 46);

    // Contiguous sum_dim
    {
        let t_e: Tensor<E, 4> = Tensor::from_data(data.clone(), &Default::default());
        let t_n: Tensor<N, 4> = Tensor::from_data(data.clone(), &Default::default());

        let result_e = t_e.sum_dim(3);
        let result_n = t_n.sum_dim(3);

        compare(
            "sum_dim3_contiguous",
            &result_e.into_data(),
            &result_n.into_data(),
        );
    }

    // Permuted sum_dim
    {
        let t_e: Tensor<E, 4> = Tensor::from_data(data.clone(), &Default::default());
        let t_n: Tensor<N, 4> = Tensor::from_data(data.clone(), &Default::default());

        let t_e = t_e.permute([0, 2, 1, 3]);
        let t_n = t_n.permute([0, 2, 1, 3]);

        let result_e = t_e.sum_dim(3);
        let result_n = t_n.sum_dim(3);

        compare(
            "sum_dim3_permuted",
            &result_e.into_data(),
            &result_n.into_data(),
        );
    }

    // Permuted mean_dim on dim 2
    {
        let t_e: Tensor<E, 4> = Tensor::from_data(data.clone(), &Default::default());
        let t_n: Tensor<N, 4> = Tensor::from_data(data.clone(), &Default::default());

        let t_e = t_e.permute([0, 2, 1, 3]);
        let t_n = t_n.permute([0, 2, 1, 3]);

        let result_e = t_e.mean_dim(2);
        let result_n = t_n.mean_dim(2);

        compare(
            "mean_dim2_permuted",
            &result_e.into_data(),
            &result_n.into_data(),
        );
    }

    // Permuted sum_dim on dim 1
    {
        let t_e: Tensor<E, 4> = Tensor::from_data(data.clone(), &Default::default());
        let t_n: Tensor<N, 4> = Tensor::from_data(data.clone(), &Default::default());

        let t_e = t_e.permute([0, 2, 1, 3]);
        let t_n = t_n.permute([0, 2, 1, 3]);

        let result_e = t_e.sum_dim(1);
        let result_n = t_n.sum_dim(1);

        compare(
            "sum_dim1_permuted",
            &result_e.into_data(),
            &result_n.into_data(),
        );
    }
}

// ============================================================================
// Test 4: Erf (element-wise) on permuted tensor
// ============================================================================

#[test]
fn test_erf_permuted() {
    eprintln!("\n=== Erf Element-wise Permuted ===");

    let data = rand_data(&[2, 4, 8, 16], 47);

    // Contiguous
    {
        let t_e: Tensor<E, 4> = Tensor::from_data(data.clone(), &Default::default());
        let t_n: Tensor<N, 4> = Tensor::from_data(data.clone(), &Default::default());

        let result_e = burn_tensor::activation::gelu(t_e);
        let result_n = burn_tensor::activation::gelu(t_n);

        compare(
            "gelu_contiguous",
            &result_e.into_data(),
            &result_n.into_data(),
        );
    }

    // Permuted
    {
        let t_e: Tensor<E, 4> = Tensor::from_data(data.clone(), &Default::default());
        let t_n: Tensor<N, 4> = Tensor::from_data(data.clone(), &Default::default());

        let t_e = t_e.permute([0, 2, 1, 3]);
        let t_n = t_n.permute([0, 2, 1, 3]);

        let result_e = burn_tensor::activation::gelu(t_e);
        let result_n = burn_tensor::activation::gelu(t_n);

        compare(
            "gelu_permuted",
            &result_e.into_data(),
            &result_n.into_data(),
        );
    }
}

// ============================================================================
// Test 5: Binary ops on permuted tensors
// ============================================================================

#[test]
fn test_binary_ops_permuted() {
    eprintln!("\n=== Binary Ops Permuted ===");

    let data_a = rand_data(&[2, 8, 4, 16], 48);
    let data_b = rand_data(&[2, 8, 4, 16], 49);

    // Add: contiguous
    {
        let a_e: Tensor<E, 4> = Tensor::from_data(data_a.clone(), &Default::default());
        let a_n: Tensor<N, 4> = Tensor::from_data(data_a.clone(), &Default::default());
        let b_e: Tensor<E, 4> = Tensor::from_data(data_b.clone(), &Default::default());
        let b_n: Tensor<N, 4> = Tensor::from_data(data_b.clone(), &Default::default());

        compare(
            "add_contiguous",
            &(a_e + b_e).into_data(),
            &(a_n + b_n).into_data(),
        );
    }

    // Add: permuted
    {
        let a_e: Tensor<E, 4> = Tensor::from_data(data_a.clone(), &Default::default());
        let a_n: Tensor<N, 4> = Tensor::from_data(data_a.clone(), &Default::default());
        let b_e: Tensor<E, 4> = Tensor::from_data(data_b.clone(), &Default::default());
        let b_n: Tensor<N, 4> = Tensor::from_data(data_b.clone(), &Default::default());

        let a_e = a_e.permute([0, 2, 1, 3]);
        let a_n = a_n.permute([0, 2, 1, 3]);
        let b_e = b_e.permute([0, 2, 1, 3]);
        let b_n = b_n.permute([0, 2, 1, 3]);

        compare(
            "add_permuted",
            &(a_e + b_e).into_data(),
            &(a_n + b_n).into_data(),
        );
    }

    // Mul: permuted
    {
        let a_e: Tensor<E, 4> = Tensor::from_data(data_a.clone(), &Default::default());
        let a_n: Tensor<N, 4> = Tensor::from_data(data_a.clone(), &Default::default());
        let b_e: Tensor<E, 4> = Tensor::from_data(data_b.clone(), &Default::default());
        let b_n: Tensor<N, 4> = Tensor::from_data(data_b.clone(), &Default::default());

        let a_e = a_e.permute([0, 2, 1, 3]);
        let a_n = a_n.permute([0, 2, 1, 3]);
        let b_e = b_e.permute([0, 2, 1, 3]);
        let b_n = b_n.permute([0, 2, 1, 3]);

        compare(
            "mul_permuted",
            &(a_e * b_e).into_data(),
            &(a_n * b_n).into_data(),
        );
    }

    // Scalar mul: permuted
    {
        let a_e: Tensor<E, 4> = Tensor::from_data(data_a.clone(), &Default::default());
        let a_n: Tensor<N, 4> = Tensor::from_data(data_a.clone(), &Default::default());

        let a_e = a_e.permute([0, 2, 1, 3]);
        let a_n = a_n.permute([0, 2, 1, 3]);

        compare(
            "scalar_mul_permuted",
            &(a_e * 0.125).into_data(),
            &(a_n * 0.125).into_data(),
        );
    }
}

// ============================================================================
// Test 6: Full attention block (the composition test)
// ============================================================================

#[test]
fn test_full_attention_block() {
    eprintln!("\n=== Full Attention Block ===");

    // Simulating multi-head attention:
    // Q, K, V start as [B, S, H*D], get reshaped to [B, S, H, D], permuted to [B, H, S, D]
    // attn = softmax(Q @ K^T / sqrt(d)) @ V
    // then permute back to [B, S, H, D], reshape to [B, S, H*D]

    let b = 2;
    let s = 8;
    let h = 4;
    let d = 16;

    let q_data = rand_data(&[b, s, h, d], 50);
    let k_data = rand_data(&[b, s, h, d], 51);
    let v_data = rand_data(&[b, s, h, d], 52);

    // Run on Flex
    let attn_e = {
        let q: Tensor<E, 4> = Tensor::from_data(q_data.clone(), &Default::default());
        let k: Tensor<E, 4> = Tensor::from_data(k_data.clone(), &Default::default());
        let v: Tensor<E, 4> = Tensor::from_data(v_data.clone(), &Default::default());

        // Permute [B,S,H,D] -> [B,H,S,D]
        let q = q.permute([0, 2, 1, 3]);
        let k = k.permute([0, 2, 1, 3]);
        let v = v.permute([0, 2, 1, 3]);

        // Q @ K^T / sqrt(d)
        let scale = 1.0 / (d as f64).sqrt();
        let scores = q.matmul(k.transpose()) * scale;

        // softmax
        let weights = burn_tensor::activation::softmax(scores, 3);

        // weights @ V
        let out = weights.matmul(v);

        // Permute back [B,H,S,D] -> [B,S,H,D]
        let out = out.permute([0, 2, 1, 3]);

        // Reshape to [B,S,H*D]
        let out: Tensor<E, 3> = out.reshape([b, s, h * d]);
        out.into_data()
    };

    // Run on NdArray
    let attn_n = {
        let q: Tensor<N, 4> = Tensor::from_data(q_data.clone(), &Default::default());
        let k: Tensor<N, 4> = Tensor::from_data(k_data.clone(), &Default::default());
        let v: Tensor<N, 4> = Tensor::from_data(v_data.clone(), &Default::default());

        let q = q.permute([0, 2, 1, 3]);
        let k = k.permute([0, 2, 1, 3]);
        let v = v.permute([0, 2, 1, 3]);

        let scale = 1.0 / (d as f64).sqrt();
        let scores = q.matmul(k.transpose()) * scale;

        let weights = burn_tensor::activation::softmax(scores, 3);

        let out = weights.matmul(v);
        let out = out.permute([0, 2, 1, 3]);
        let out: Tensor<N, 3> = out.reshape([b, s, h * d]);
        out.into_data()
    };

    let e_vals: Vec<f32> = attn_e.to_vec().unwrap();
    let n_vals: Vec<f32> = attn_n.to_vec().unwrap();

    let mut max_diff: f32 = 0.0;
    let mut sum_diff: f32 = 0.0;
    for (&e, &n) in e_vals.iter().zip(n_vals.iter()) {
        let diff = (e - n).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        sum_diff += diff;
    }
    let mean_diff = sum_diff / e_vals.len() as f32;

    let status = if max_diff > 1e-4 { "FAIL" } else { "ok" };
    eprintln!("  {status:>4} full_attention: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}");

    assert!(
        max_diff < 1e-4,
        "Full attention block diff too large: max={max_diff:.2e}"
    );
}

// ============================================================================
// Test 7: Reshape after permute (forces to_contiguous)
// ============================================================================

#[test]
fn test_reshape_after_permute() {
    eprintln!("\n=== Reshape After Permute ===");

    let data = rand_data(&[2, 8, 4, 16], 53);

    // Permute then reshape (exercises to_contiguous path)
    let t_e: Tensor<E, 4> = Tensor::from_data(data.clone(), &Default::default());
    let t_n: Tensor<N, 4> = Tensor::from_data(data.clone(), &Default::default());

    // [2,8,4,16] -> permute [0,2,1,3] -> [2,4,8,16] -> reshape [2,4,128]
    let t_e = t_e.permute([0, 2, 1, 3]);
    let t_n = t_n.permute([0, 2, 1, 3]);

    let r_e: Tensor<E, 3> = t_e.reshape([2, 4, 128]);
    let r_n: Tensor<N, 3> = t_n.reshape([2, 4, 128]);

    compare("reshape_after_permute", &r_e.into_data(), &r_n.into_data());
}

// ============================================================================
// Test 8: Matmul where one operand is permuted and other is contiguous
// ============================================================================

#[test]
fn test_matmul_mixed_contiguity() {
    eprintln!("\n=== Matmul Mixed Contiguity ===");

    let b = 2;
    let h = 4;
    let s = 8;
    let d = 16;

    // Weights are contiguous [B,H,D,D], input is permuted
    let input_data = rand_data(&[b, s, h, d], 54);
    let weight_data = rand_data(&[b, h, d, d], 55);

    let i_e: Tensor<E, 4> = Tensor::from_data(input_data.clone(), &Default::default());
    let i_n: Tensor<N, 4> = Tensor::from_data(input_data.clone(), &Default::default());
    let w_e: Tensor<E, 4> = Tensor::from_data(weight_data.clone(), &Default::default());
    let w_n: Tensor<N, 4> = Tensor::from_data(weight_data.clone(), &Default::default());

    // Permute input only: [B,S,H,D] -> [B,H,S,D]
    let i_e = i_e.permute([0, 2, 1, 3]);
    let i_n = i_n.permute([0, 2, 1, 3]);

    // Matmul: [B,H,S,D] @ [B,H,D,D] -> [B,H,S,D]
    let result_e = i_e.matmul(w_e);
    let result_n = i_n.matmul(w_n);

    compare(
        "matmul_permuted_x_contiguous",
        &result_e.into_data(),
        &result_n.into_data(),
    );
}

// ============================================================================
// Test 9: Layernorm-like pattern on permuted tensor
// (mean_dim + sub + mul + sum_dim + div)
// ============================================================================

#[test]
fn test_layernorm_pattern_permuted() {
    eprintln!("\n=== Layernorm Pattern Permuted ===");

    let data = rand_data(&[2, 8, 4, 16], 56);

    // Manual layernorm on last dim of a permuted [2,4,8,16] tensor
    let run = |name: &str, permute: bool| {
        let t_e: Tensor<E, 4> = Tensor::from_data(data.clone(), &Default::default());
        let t_n: Tensor<N, 4> = Tensor::from_data(data.clone(), &Default::default());

        let (t_e, t_n) = if permute {
            (t_e.permute([0, 2, 1, 3]), t_n.permute([0, 2, 1, 3]))
        } else {
            (t_e, t_n)
        };

        // mean
        let mean_e = t_e.clone().mean_dim(3);
        let mean_n = t_n.clone().mean_dim(3);

        // x - mean
        let centered_e = t_e.clone() - mean_e;
        let centered_n = t_n.clone() - mean_n;

        // variance = mean((x - mean)^2)
        let var_e = (centered_e.clone() * centered_e.clone()).mean_dim(3);
        let var_n = (centered_n.clone() * centered_n.clone()).mean_dim(3);

        // normalize: (x - mean) / sqrt(var + eps)
        let eps = 1e-5;
        let norm_e = centered_e / (var_e + eps).sqrt();
        let norm_n = centered_n / (var_n + eps).sqrt();

        compare(name, &norm_e.into_data(), &norm_n.into_data());
    };

    run("layernorm_contiguous", false);
    run("layernorm_permuted", true);
}
