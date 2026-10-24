use super::*;
use burn_tensor::activation::softmax;
use burn_tensor::module::attention;
use burn_tensor::ops::AttentionModuleOptions;
use burn_tensor::{Bool, Distribution, Tensor, TensorData, Tolerance};

/// Attention gradients match the same computation spelled out as tensor ops, for every
/// option. `sq != sk` pins the causal mask to the bottom-right corner.
#[test]
fn attention_grads_match_composed_ops() {
    let device = AutodiffDevice::new();
    let (b, h, sq, sk, e) = (2, 2, 5, 6, 4);
    let random = |shape: [usize; 4]| {
        TestTensor::<4>::random(shape, Distribution::Default, &device).into_data()
    };
    let (q_data, k_data, v_data) = (
        random([b, h, sq, e]),
        random([b, h, sk, e]),
        random([b, h, sk, e]),
    );
    let bias_data = random([b, h, sq, sk]);
    let mask_data = TensorData::new(
        (0..b * h * sq * sk).map(|i| i % 5 == 3).collect::<Vec<_>>(),
        [b, h, sq, sk],
    );
    let causal_data = TensorData::new(
        (0..sq * sk)
            .map(|i| i % sk > i / sk + (sk - sq))
            .collect::<Vec<_>>(),
        [1, 1, sq, sk],
    );

    for (is_causal, softcap, masked, scale) in [
        (false, None, false, None),
        (true, None, false, None),
        (false, Some(2.0), false, None),
        (false, None, true, Some(0.3)),
        (true, Some(5.0), true, None),
    ] {
        let run = |composed: bool| {
            let q = TestTensor::<4>::from_data(q_data.clone(), &device).require_grad();
            let k = TestTensor::<4>::from_data(k_data.clone(), &device).require_grad();
            let v = TestTensor::<4>::from_data(v_data.clone(), &device).require_grad();
            let bias = TestTensor::<4>::from_data(bias_data.clone(), &device).require_grad();
            let mask = Tensor::<4, Bool>::from_data(mask_data.clone(), &device);

            let output = if composed {
                let scale = scale.unwrap_or(1.0 / (e as f64).sqrt());
                let mut scores = q
                    .clone()
                    .matmul(k.clone().swap_dims(2, 3))
                    .mul_scalar(scale);
                if let Some(cap) = softcap {
                    scores = scores.div_scalar(cap).tanh().mul_scalar(cap);
                }
                if masked {
                    scores = (scores + bias.clone()).mask_fill(mask, f32::NEG_INFINITY);
                }
                if is_causal {
                    let causal = Tensor::<4, Bool>::from_data(causal_data.clone(), &device)
                        .expand([b, h, sq, sk]);
                    scores = scores.mask_fill(causal, f32::NEG_INFINITY);
                }
                softmax(scores, 3).matmul(v.clone())
            } else {
                let options = AttentionModuleOptions {
                    scale,
                    softcap,
                    is_causal,
                    ..Default::default()
                };
                let bias = masked.then(|| bias.clone());
                attention(
                    q.clone(),
                    k.clone(),
                    v.clone(),
                    masked.then_some(mask),
                    bias,
                    options,
                )
            };

            let weights = TensorData::new(
                (0..output.shape().num_elements())
                    .map(|i| (i % 5) as f32 - 2.0)
                    .collect::<Vec<_>>(),
                output.shape(),
            );
            let grads = (output.clone() * TestTensor::from_data(weights, &device))
                .sum()
                .backward();
            let mut results = vec![
                output.into_data(),
                q.grad(&grads).unwrap().into_data(),
                k.grad(&grads).unwrap().into_data(),
                v.grad(&grads).unwrap().into_data(),
            ];
            if masked {
                results.push(bias.grad(&grads).unwrap().into_data());
            }
            results
        };

        for (actual, expected) in run(false).into_iter().zip(run(true)) {
            actual.assert_approx_eq::<FloatElem>(&expected, Tolerance::permissive());
        }
    }
}

/// Grouped-query attention gradients match those of attention over K/V repeated per query
/// head: each K/V head's gradient sums over the query heads sharing it.
#[test]
fn attention_grouped_query_grads_match_repeated_kv() {
    let device = AutodiffDevice::new();
    let (b, q_heads, kv_heads, sq, sk, e) = (2, 6, 2, 4, 7, 8);
    let groups = q_heads / kv_heads;
    let random = |shape: [usize; 4]| {
        TestTensor::<4>::random(shape, Distribution::Default, &device).into_data()
    };
    let (q_data, k_data, v_data) = (
        random([b, q_heads, sq, e]),
        random([b, kv_heads, sk, e]),
        random([b, kv_heads, sk, e]),
    );

    for is_causal in [false, true] {
        let run = |repeated: bool| {
            let q = TestTensor::<4>::from_data(q_data.clone(), &device).require_grad();
            let k = TestTensor::<4>::from_data(k_data.clone(), &device).require_grad();
            let v = TestTensor::<4>::from_data(v_data.clone(), &device).require_grad();
            let repeat = |x: TestTensor<4>| {
                x.unsqueeze_dim::<5>(2)
                    .expand([b, kv_heads, groups, sk, e])
                    .reshape([b, q_heads, sk, e])
            };
            let (k_in, v_in) = if repeated {
                (repeat(k.clone()), repeat(v.clone()))
            } else {
                (k.clone(), v.clone())
            };
            let options = AttentionModuleOptions {
                is_causal,
                softcap: Some(5.0),
                ..Default::default()
            };
            let output = attention(q.clone(), k_in, v_in, None, None, options);
            let grads = (output.clone() * output.clone()).sum().backward();
            [
                output.inner(),
                q.grad(&grads).unwrap(),
                k.grad(&grads).unwrap(),
                v.grad(&grads).unwrap(),
            ]
        };

        for (actual, expected) in run(false).into_iter().zip(run(true)) {
            actual
                .into_data()
                .assert_approx_eq::<FloatElem>(&expected.into_data(), Tolerance::permissive());
        }
    }
}
