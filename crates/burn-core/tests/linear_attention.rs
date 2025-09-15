use burn_core::nn::attention::{LinearAttnInput, LinearAttentionConfig};
use burn_core::prelude::Backend;
use burn_core::tensor::{Distribution, Shape, Tensor};

type TB = burn_ndarray::NdArray<f32>;

#[test]
fn linear_attention_self_shapes() {
    let device = <TB as Backend>::Device::default();
    let d_model = 32;
    let n_heads = 4;

    let la = LinearAttentionConfig::new(d_model, n_heads).init::<TB>(&device);
    let x = Tensor::<TB, 3>::random([2, 16, d_model], Distribution::Default, &device);
    let out = la.forward(LinearAttnInput::self_attn(x));
    assert_eq!(out.context.shape(), Shape::new([2, 16, d_model]));
}

