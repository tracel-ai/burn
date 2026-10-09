use burn_tensor::{
    Device, Distribution, GroupPlacement,
    GroupPlacement::{Partial, Replicated, Sharded},
    Tensor, TensorData, Tolerance, activation, module,
};

#[test]
fn megatron_mlp_on_a_device_group_matches_one_device() {
    for members in [1, 2, 3, 4] {
        let reference = mlp(&Device::flex().autodiff(), [Replicated; 5]);
        let split = mlp(
            &group(members).autodiff(),
            [
                Replicated,
                Sharded { dim: 1 },
                Sharded { dim: 0 },
                Sharded { dim: 0 },
                Replicated,
            ],
        );

        assert_eq!(
            split.grad_placements(),
            [
                Some(Partial),
                Some(Sharded { dim: 1 }),
                Some(Sharded { dim: 0 }),
                Some(Sharded { dim: 0 }),
                Some(Replicated)
            ],
            "{members} members"
        );
        split.assert_matches(reference);
    }
}

#[test]
fn megatron_attention_on_a_device_group_matches_one_device() {
    for members in [1, 2, 4] {
        let reference = attention(&Device::flex().autodiff(), [Replicated; 4]);
        let split = attention(
            &group(members).autodiff(),
            [
                Sharded { dim: 1 },
                Sharded { dim: 1 },
                Sharded { dim: 1 },
                Sharded { dim: 0 },
            ],
        );

        assert_eq!(
            split.grad_placements()[1..],
            [
                Some(Sharded { dim: 1 }),
                Some(Sharded { dim: 1 }),
                Some(Sharded { dim: 1 }),
                Some(Sharded { dim: 0 })
            ],
            "{members} members"
        );
        split.assert_matches(reference);
    }
}

#[test]
fn a_tensor_moved_onto_a_group_is_replicated_and_moves_back_whole() {
    let data = values([4, 6], 1);
    let tensor = Tensor::<2>::from_data(data.clone(), &Device::flex()).to_device(&group(2));
    assert_eq!(tensor.placement(), Some(Replicated));

    let back = tensor.place(Sharded { dim: 1 }).to_device(&Device::flex());
    assert_eq!(back.placement(), None);
    back.into_data().assert_approx_eq::<f32>(&data, tolerance());
}

#[test]
fn a_tensor_placed_mid_graph_passes_its_gradient_through() {
    let device = group(2).autodiff();
    let x = Tensor::<1>::from_data(values([4], 1), &device).require_grad();
    let y = x.clone().mul_scalar(2.0).place(Sharded { dim: 0 }) + x.clone();
    let grads = y.sum().backward();

    let grad = x.grad(&grads).expect("x has a gradient");
    grad.into_data()
        .assert_approx_eq::<f32>(&TensorData::from([3.0f32; 4]), tolerance());
}

#[test]
fn a_random_tensor_is_the_same_on_every_member() {
    let device = group(2);
    let random = Tensor::<2>::random([4, 6], Distribution::Default, &device);
    let shifted = random.clone() + Tensor::ones([4, 6], &device);

    let expected = random
        .to_device(&Device::flex())
        .add_scalar(1.0)
        .into_data();
    let from_every_member = shifted.place(Sharded { dim: 1 }).to_device(&Device::flex());
    from_every_member
        .into_data()
        .assert_approx_eq::<f32>(&expected, tolerance());
}

#[test]
fn a_vocab_split_embedding_keeps_its_gradient_split() {
    let indices = TensorData::from([[0i64, 6, 3], [2, 2, 5]]);
    let run = |device: &Device, placement| {
        let weights = leaf::<2>(values([7, 4], 1), device, placement);
        let ids = Tensor::<2, burn_tensor::Int>::from_data(indices.clone(), device);
        let output = module::embedding(weights.clone(), ids);
        let scale = Tensor::<3>::from_data(values([2, 3, 4], 2), device);
        let grads = (output.clone() * scale).mean().backward();
        Run {
            output: output.inner().into_data(),
            grads: vec![Grad::new(weights.grad(&grads))],
        }
    };
    let reference = run(&Device::flex().autodiff(), Replicated);
    for members in [2, 3] {
        let split = run(&group(members).autodiff(), Sharded { dim: 0 });

        assert_eq!(
            split.grad_placements(),
            [Some(Sharded { dim: 0 })],
            "{members} members"
        );
        split.assert_matches(Run {
            output: reference.output.clone(),
            grads: vec![Grad {
                placement: None,
                data: reference.grads[0].data.clone(),
            }],
        });
    }
}

/// What a forward and backward pass produced: the output, then each leaf's gradient.
struct Run {
    output: TensorData,
    grads: Vec<Grad>,
}

struct Grad {
    placement: Option<GroupPlacement>,
    data: TensorData,
}

impl Grad {
    fn new<const D: usize>(grad: Option<Tensor<D>>) -> Self {
        let grad = grad.expect("Every leaf has a gradient");
        Self {
            placement: grad.placement(),
            data: grad.into_data(),
        }
    }
}

impl Run {
    fn grad_placements(&self) -> Vec<Option<GroupPlacement>> {
        self.grads.iter().map(|grad| grad.placement).collect()
    }

    fn assert_matches(self, expected: Run) {
        self.output
            .assert_approx_eq::<f32>(&expected.output, tolerance());
        for (actual, expected) in self.grads.into_iter().zip(expected.grads) {
            actual
                .data
                .assert_approx_eq::<f32>(&expected.data, tolerance());
        }
    }
}

/// Placements of x, w1, b1, w2 and b2.
fn mlp(device: &Device, [px, pw1, pb1, pw2, pb2]: [GroupPlacement; 5]) -> Run {
    const BATCH: usize = 4;
    const D_IN: usize = 6;
    const HIDDEN: usize = 12;
    const D_OUT: usize = 5;

    let x = leaf::<2>(values([BATCH, D_IN], 1), device, px);
    let w1 = leaf::<2>(values([D_IN, HIDDEN], 2), device, pw1);
    let b1 = leaf::<1>(values([HIDDEN], 3), device, pb1);
    let w2 = leaf::<2>(values([HIDDEN, D_OUT], 4), device, pw2);
    let b2 = leaf::<1>(values([D_OUT], 5), device, pb2);

    let hidden = activation::relu(module::linear(x.clone(), w1.clone(), Some(b1.clone())));
    let output = module::linear(hidden, w2.clone(), Some(b2.clone()));
    let target = Tensor::from_data(values([BATCH, D_OUT], 6), device);
    let grads = (output.clone() - target).square().mean().backward();
    Run {
        output: output.inner().into_data(),
        grads: vec![
            Grad::new(x.grad(&grads)),
            Grad::new(w1.grad(&grads)),
            Grad::new(b1.grad(&grads)),
            Grad::new(w2.grad(&grads)),
            Grad::new(b2.grad(&grads)),
        ],
    }
}

/// Placements of the query, key, value and output weights.
fn attention(device: &Device, [pq, pk, pv, po]: [GroupPlacement; 4]) -> Run {
    const BATCH: usize = 2;
    const SEQ: usize = 5;
    const HEADS: usize = 4;
    const HEAD_DIM: usize = 3;
    const EMBED: usize = HEADS * HEAD_DIM;

    let x = leaf::<3>(values([BATCH, SEQ, EMBED], 1), device, Replicated);
    let weight = |seed, placement| leaf::<2>(values([EMBED, EMBED], seed), device, placement);
    let (wq, wk, wv, wo) = (weight(2, pq), weight(3, pk), weight(4, pv), weight(5, po));

    let heads = |weight: &Tensor<2>| {
        module::linear(x.clone(), weight.clone(), None)
            .reshape([BATCH, SEQ, HEADS, HEAD_DIM])
            .swap_dims(1, 2)
    };
    let (q, k, v) = (heads(&wq), heads(&wk), heads(&wv));
    let scores = q
        .matmul(k.swap_dims(2, 3))
        .div_scalar((HEAD_DIM as f64).sqrt());
    let context = activation::softmax(scores, 3)
        .matmul(v)
        .swap_dims(1, 2)
        .reshape([BATCH, SEQ, EMBED]);
    let output = module::linear(context, wo.clone(), None) + x.clone();
    let grads = output.clone().square().mean().backward();
    Run {
        output: output.inner().into_data(),
        grads: vec![
            Grad::new(x.grad(&grads)),
            Grad::new(wq.grad(&grads)),
            Grad::new(wk.grad(&grads)),
            Grad::new(wv.grad(&grads)),
            Grad::new(wo.grad(&grads)),
        ],
    }
}

fn group(members: usize) -> Device {
    let devices: Vec<Device> = member_devices().into_iter().cycle().take(members).collect();
    Device::group(&devices)
}

#[cfg(feature = "cuda")]
fn member_devices() -> Vec<Device> {
    Device::enumerate(burn_tensor::DeviceType::Cuda).to_vec()
}

#[cfg(not(feature = "cuda"))]
fn member_devices() -> Vec<Device> {
    vec![Device::flex()]
}

/// A leaf on `device`, placed when the device is a group.
fn leaf<const D: usize>(data: TensorData, device: &Device, placement: GroupPlacement) -> Tensor<D> {
    let tensor = Tensor::from_data(data, device);
    match tensor.placement() {
        Some(_) => tensor.place(placement).require_grad(),
        None => tensor.require_grad(),
    }
}

fn values<const D: usize>(shape: [usize; D], seed: usize) -> TensorData {
    let count = shape.iter().product::<usize>();
    let values: Vec<f32> = (0..count)
        .map(|i| ((i * 7 + seed * 13) as f32 * 0.37).sin())
        .collect();
    TensorData::new(values, shape)
}

#[cfg(not(feature = "cuda"))]
fn tolerance() -> Tolerance<f32> {
    Tolerance::rel_abs(1e-4, 1e-5)
}

/// CUDA rounds an f32 matmul's operands through TF32 for some shapes, and a split changes them.
#[cfg(feature = "cuda")]
fn tolerance() -> Tolerance<f32> {
    Tolerance::permissive()
}
