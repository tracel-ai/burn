use burn_core::nn::RotaryEncodingConfig;
use burn_core::nn::attention::{
    AttnWindow, StreamingMhaCache, StreamingMultiHeadAttentionConfig, StreamingParams,
};
use burn_core::tensor::{Distribution, Shape, Tensor};
type TB = burn_ndarray::NdArray<f32>;
use burn_tensor::Tolerance;
use burn_tensor::ops::FloatElem;

#[test]
fn streaming_no_window_vs_full_window_equal() {
    let device = Default::default();
    let b = 2;
    let t = 12;
    let d_model = 32;
    let n_heads = 4;

    let x = Tensor::<TB, 3>::random([b, t, d_model], Distribution::Default, &device);

    let smha = StreamingMultiHeadAttentionConfig::new(d_model, n_heads)
        .with_dropout(0.0)
        .init::<TB>(&device);
    let mut cache1 = StreamingMhaCache::new(
        &device,
        b,
        /*cache_len*/ 64,
        n_heads,
        d_model / n_heads,
        /*sink*/ 0,
    );
    let out1 = smha.forward_streaming(
        x.clone(),
        &mut cache1,
        StreamingParams {
            rope: None,
            start_pos: 0,
            window: AttnWindow::Full,
        },
    );

    let mut cache2 = StreamingMhaCache::new(
        &device,
        b,
        /*cache_len*/ 64,
        n_heads,
        d_model / n_heads,
        /*sink*/ 0,
    );
    let out2 = smha.forward_streaming(
        x,
        &mut cache2,
        StreamingParams {
            rope: None,
            start_pos: 0,
            window: AttnWindow::Window(t),
        },
    );

    assert_eq!(out1.shape(), Shape::new([b, t, d_model]));
    out1.into_data()
        .assert_approx_eq::<FloatElem<TB>>(&out2.into_data(), Tolerance::default());
}

#[test]
fn streaming_chunked_with_rope_matches_full_call() {
    let device = Default::default();
    let b = 1;
    let t = 16;
    let d_model = 32;
    let n_heads = 4;

    let x = Tensor::<TB, 3>::random([b, t, d_model], Distribution::Default, &device);
    let head_dim = d_model / n_heads;
    let rope = RotaryEncodingConfig::new(512, head_dim).init::<TB>(&device);
    let smha = StreamingMultiHeadAttentionConfig::new(d_model, n_heads)
        .with_dropout(0.0)
        .init::<TB>(&device);

    // Full single-shot (one chunk)
    let mut cache_full = StreamingMhaCache::new(
        &device, b, /*cache_len*/ 64, n_heads, head_dim, /*sink*/ 0,
    );
    let out_full = smha.forward_streaming(
        x.clone(),
        &mut cache_full,
        StreamingParams {
            rope: Some(&rope),
            start_pos: 0,
            window: AttnWindow::Window(t),
        },
    );

    // Chunked
    let mut cache_chunked = StreamingMhaCache::new(
        &device, b, /*cache_len*/ 64, n_heads, head_dim, /*sink*/ 0,
    );
    let mut outputs = Vec::new();
    let chunk = 4;
    for i in 0..(t / chunk) {
        let start = i * chunk;
        let x_i = x.clone().slice([0..b, start..start + chunk, 0..d_model]);
        let params = StreamingParams {
            rope: Some(&rope),
            start_pos: start,
            window: AttnWindow::Window(t),
        };
        let y = smha.forward_streaming(x_i, &mut cache_chunked, params);
        outputs.push(y);
    }
    let out_chunked = Tensor::cat(outputs, 1);

    assert_eq!(out_full.shape(), Shape::new([b, t, d_model]));
    out_full
        .into_data()
        .assert_approx_eq::<FloatElem<TB>>(&out_chunked.into_data(), Tolerance::rel_abs(0.5, 0.2));
}
