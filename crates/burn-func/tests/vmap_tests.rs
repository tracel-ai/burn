use burn_autodiff::Autodiff;
use burn_backend::{Backend, TensorData};
use burn_flex::{Flex, FlexDevice};
use burn_func::{FuncTensor, Vmap, VmapAxis, vmap, vmap_with_axes};

fn assert_backend<B: Backend>() {}

#[test]
fn vmap_and_autodiff_compose_at_the_type_level() {
    assert_backend::<Vmap<Autodiff<Flex>>>();
    assert_backend::<Autodiff<Vmap<Flex>>>();
}

fn matrix(values: [[f32; 2]; 2]) -> FuncTensor<Flex, 2> {
    FuncTensor::from_data(TensorData::from(values), &FlexDevice)
}

#[test]
fn vmap_matches_loop_for_elementwise() {
    let x = matrix([[1.0, 2.0], [3.0, 4.0]]);
    let f = |x: &FuncTensor<Vmap<Flex>, 2>| (x.clone() * 2.0f64 + 1.0f64).sum();
    let out: FuncTensor<Flex, 1> = vmap::<Flex, _, _, _>(f)(&x);

    let values = out.into_data_sync().try_to_vec::<f32>().unwrap();
    assert_eq!(values, vec![8.0, 16.0]);
}

#[test]
fn vmap_sum_and_mean_reduce_only_logical_axes() {
    let x = matrix([[1.0, 2.0], [3.0, 4.0]]);

    let sum = |x: &FuncTensor<Vmap<Flex>, 2>| x.clone().sum();
    let sum_out: FuncTensor<Flex, 1> = vmap::<Flex, _, _, _>(sum)(&x);
    assert_eq!(
        sum_out.into_data_sync().try_to_vec::<f32>().unwrap(),
        vec![3.0, 7.0]
    );

    let mean = |x: &FuncTensor<Vmap<Flex>, 2>| x.clone().mean();
    let mean_out: FuncTensor<Flex, 1> = vmap::<Flex, _, _, _>(mean)(&x);
    assert_eq!(
        mean_out.into_data_sync().try_to_vec::<f32>().unwrap(),
        vec![1.5, 3.5]
    );
}

#[test]
fn vmap_sum_dim_and_max_dim() {
    let x = matrix([[1.0, 4.0], [3.0, 2.0]]);

    let sum_dim = |x: &FuncTensor<Vmap<Flex>, 2>| x.clone().sum_dim(0);
    let sum_out: FuncTensor<Flex, 2> = vmap::<Flex, _, _, _>(sum_dim)(&x);
    assert_eq!(
        sum_out.into_data_sync().try_to_vec::<f32>().unwrap(),
        vec![5.0, 5.0]
    );

    let max_dim = |x: &FuncTensor<Vmap<Flex>, 2>| x.clone().max_dim(0);
    let max_out: FuncTensor<Flex, 2> = vmap::<Flex, _, _, _>(max_dim)(&x);
    assert_eq!(
        max_out.into_data_sync().try_to_vec::<f32>().unwrap(),
        vec![4.0, 3.0]
    );
}

#[test]
fn vmap_relu_and_softmax() {
    let x = matrix([[-1.0, 2.0], [3.0, -4.0]]);

    let relu = |x: &FuncTensor<Vmap<Flex>, 2>| x.clone().relu();
    let relu_out: FuncTensor<Flex, 2> = vmap::<Flex, _, _, _>(relu)(&x);
    assert_eq!(
        relu_out.into_data_sync().try_to_vec::<f32>().unwrap(),
        vec![0.0, 2.0, 3.0, 0.0]
    );

    let softmax = |x: &FuncTensor<Vmap<Flex>, 2>| x.clone().softmax(0);
    let softmax_out: FuncTensor<Flex, 2> = vmap::<Flex, _, _, _>(softmax)(&x);
    let values = softmax_out.into_data_sync().try_to_vec::<f32>().unwrap();
    for row in values.chunks(2) {
        assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }
}

#[test]
fn vmap_matmul_with_unbatched_weight() {
    let x = matrix([[1.0, 2.0], [3.0, 4.0]]);
    let weight = matrix([[1.0, 0.0], [0.0, 1.0]]);

    let f = |(x, weight): &(FuncTensor<Vmap<Flex>, 2>, FuncTensor<Vmap<Flex>, 2>)| {
        x.clone().matmul(weight.clone())
    };
    let out: FuncTensor<Flex, 2> = vmap_with_axes::<Flex, _, _, _>(
        f,
        vec![VmapAxis::Batched(0), VmapAxis::Unbatched],
    )(&(x, weight));

    let values = out.into_data_sync().try_to_vec::<f32>().unwrap();
    assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
#[allow(clippy::type_complexity)]
fn vmap_linear_layer() {
    let x = FuncTensor::<Flex, 3>::from_data(
        TensorData::from([[[1.0f32, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
        &FlexDevice,
    );
    let weight = matrix([[2.0, 0.0], [0.0, 2.0]]);
    let bias = FuncTensor::<Flex, 1>::from_data(TensorData::from([1.0f32, -1.0]), &FlexDevice);

    let f = |(x, weight, bias): &(
        FuncTensor<Vmap<Flex>, 3>,
        FuncTensor<Vmap<Flex>, 2>,
        FuncTensor<Vmap<Flex>, 1>,
    )| x.clone().linear(weight.clone(), Some(bias.clone()));

    let out: FuncTensor<Flex, 3> = vmap_with_axes::<Flex, _, _, _>(
        f,
        vec![
            VmapAxis::Batched(0),
            VmapAxis::Unbatched,
            VmapAxis::Unbatched,
        ],
    )(&(x, weight, bias));

    let values = out.into_data_sync().try_to_vec::<f32>().unwrap();
    assert_eq!(values, vec![3.0, 3.0, 7.0, 7.0, 11.0, 11.0, 15.0, 15.0]);
}

#[test]
fn vmap_cat_pad_and_slice() {
    let x = matrix([[1.0, 2.0], [3.0, 4.0]]);

    let cat = |x: &FuncTensor<Vmap<Flex>, 2>| FuncTensor::cat(vec![x.clone(), x.clone()], 0);
    let cat_out: FuncTensor<Flex, 2> = vmap::<Flex, _, _, _>(cat)(&x);
    assert_eq!(
        cat_out.into_data_sync().try_to_vec::<f32>().unwrap(),
        vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]
    );

    let pad = |x: &FuncTensor<Vmap<Flex>, 2>| {
        x.clone()
            .pad(&[(1, 1)], burn_std::ops::PadMode::Constant(0.0f32))
    };
    let pad_out: FuncTensor<Flex, 2> = vmap::<Flex, _, _, _>(pad)(&x);
    assert_eq!(
        pad_out.into_data_sync().try_to_vec::<f32>().unwrap(),
        vec![0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0]
    );

    let slice =
        |x: &FuncTensor<Vmap<Flex>, 2>| x.clone().slice(&[burn_std::Slice::new(0, Some(1), 1)]);
    let slice_out: FuncTensor<Flex, 2> = vmap::<Flex, _, _, _>(slice)(&x);
    assert_eq!(
        slice_out.into_data_sync().try_to_vec::<f32>().unwrap(),
        vec![1.0, 3.0]
    );
}

#[test]
fn vmap_reshape_and_slice_assign() {
    let x = matrix([[1.0, 2.0], [3.0, 4.0]]);

    let reshape = |x: &FuncTensor<Vmap<Flex>, 2>| x.clone().reshape::<2, _>([1, 2]);
    let reshape_out: FuncTensor<Flex, 2> = vmap::<Flex, _, _, _>(reshape)(&x);
    assert_eq!(
        reshape_out.into_data_sync().try_to_vec::<f32>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );

    let assign = |x: &FuncTensor<Vmap<Flex>, 2>| {
        let value = FuncTensor::<Vmap<Flex>, 1>::ones([1], &FlexDevice);
        x.clone()
            .slice_assign(&[burn_std::Slice::new(0, Some(1), 1)], value)
    };
    let assign_out: FuncTensor<Flex, 2> = vmap::<Flex, _, _, _>(assign)(&x);
    assert_eq!(
        assign_out.into_data_sync().try_to_vec::<f32>().unwrap(),
        vec![1.0, 2.0, 1.0, 4.0]
    );
}

#[test]
fn vmap_int_prod_primitive() {
    use burn_backend::ops::IntTensorOps;
    use burn_func::vmap::BatchedPrimitive;

    let prim: burn_backend::tensor::IntTensor<Flex> =
        Flex::int_from_data(TensorData::from([[2i32, 3], [4, 5]]), &FlexDevice);
    let batched = BatchedPrimitive::batched(prim, 2);
    let out = Vmap::<Flex>::int_prod(batched);
    let data = burn_backend::try_read_sync(Vmap::<Flex>::int_into_data(out))
        .unwrap()
        .unwrap();
    assert_eq!(data.try_to_vec::<i32>().unwrap(), vec![6, 20]);
}

#[test]
#[should_panic(expected = "bool_argwhere is not supported on batched (vmap) tensors")]
fn vmap_bool_argwhere_panics_on_batched() {
    use burn_backend::ops::BoolTensorOps;
    use burn_func::vmap::BatchedPrimitive;

    let mask: burn_backend::tensor::BoolTensor<Flex> = Flex::bool_from_data(
        TensorData::from([[true, false], [false, true]]),
        &FlexDevice,
    );
    let batched = BatchedPrimitive::batched(mask, 2);
    let _ = burn_std::future::block_on(Vmap::<Flex>::bool_argwhere(
        batched,
        burn_backend::IntDType::I32,
    ));
}

#[test]
#[allow(clippy::type_complexity)]
fn vmap_over_grad_per_sample_weights() {
    use burn_autodiff::Autodiff;
    use burn_backend::AutodiffBackend;
    use burn_func::grad;

    let w0 = FuncTensor::<Flex, 1>::from_data(TensorData::from([0.5f32]), &FlexDevice);
    let xs = [[1.0f32], [2.0], [3.0]];
    let ys = [[2.0f32], [4.0], [6.0]];
    let x_batch = FuncTensor::<Flex, 2>::from_data(TensorData::from(xs), &FlexDevice);
    let y_batch = FuncTensor::<Flex, 2>::from_data(TensorData::from(ys), &FlexDevice);

    let f = |(w, x, y): &(
        FuncTensor<Vmap<Flex>, 1>,
        FuncTensor<Vmap<Flex>, 2>,
        FuncTensor<Vmap<Flex>, 2>,
    )| {
        let x1: FuncTensor<Vmap<Flex>, 1> = x.clone().reshape([1]);
        let y1: FuncTensor<Vmap<Flex>, 1> = y.clone().reshape([1]);
        let loss = |w: &FuncTensor<Autodiff<Vmap<Flex>>, 1>| {
            let x_ad = FuncTensor::<Autodiff<Vmap<Flex>>, 1>::from_primitive(
                Autodiff::<Vmap<Flex>>::from_inner(x1.primitive().clone()),
            );
            let y_ad = FuncTensor::<Autodiff<Vmap<Flex>>, 1>::from_primitive(
                Autodiff::<Vmap<Flex>>::from_inner(y1.primitive().clone()),
            );
            let pred = w.clone() * x_ad;
            (pred - y_ad).powf_scalar(2.0.into()).mean()
        };
        grad::<Vmap<Flex>, _, _>(loss)(w)
    };

    let vmap_grads: FuncTensor<Flex, 1> = vmap_with_axes::<Flex, _, _, _>(
        f,
        vec![
            VmapAxis::Unbatched,
            VmapAxis::Batched(0),
            VmapAxis::Batched(0),
        ],
    )(&(w0.clone(), x_batch, y_batch));

    let vmap_vals = vmap_grads.into_data_sync().try_to_vec::<f32>().unwrap();

    let w0_val = 0.5f32;
    let expected: Vec<f32> = xs
        .iter()
        .zip(ys.iter())
        .map(|(x, y)| 2.0 * (w0_val * x[0] - y[0]) * x[0])
        .collect();

    // Also compute via per-sample loop
    let mut loop_grads = Vec::new();
    for i in 0..3 {
        let xi = FuncTensor::<Flex, 1>::from_data(TensorData::from([xs[i][0]]), &FlexDevice);
        let yi = FuncTensor::<Flex, 1>::from_data(TensorData::from([ys[i][0]]), &FlexDevice);
        let sample_loss = |w: &FuncTensor<Autodiff<Flex>, 1>| {
            let xi_ad = FuncTensor::<Autodiff<Flex>, 1>::from_primitive(
                Autodiff::<Flex>::from_inner(xi.primitive().clone()),
            );
            let yi_ad = FuncTensor::<Autodiff<Flex>, 1>::from_primitive(
                Autodiff::<Flex>::from_inner(yi.primitive().clone()),
            );
            let pred = w.clone() * xi_ad;
            (pred - yi_ad).powf_scalar(2.0.into()).mean()
        };
        let g = grad::<Flex, _, _>(sample_loss)(&w0);
        loop_grads.push(g.into_data_sync().try_to_vec::<f32>().unwrap()[0]);
    }

    assert_eq!(vmap_vals.len(), 3);
    for i in 0..3 {
        assert!(
            (vmap_vals[i] - expected[i]).abs() < 1e-5,
            "vmap_vals[{i}] = {}, expected = {}",
            vmap_vals[i],
            expected[i]
        );
        assert!(
            (vmap_vals[i] - loop_grads[i]).abs() < 1e-5,
            "vmap_vals[{i}] = {}, loop_grads[{i}] = {}",
            vmap_vals[i],
            loop_grads[i]
        );
    }
}

#[test]
fn vmap_rank_broadcast_binary_ops() {
    let device = FlexDevice;
    // Batch size N = 4. Logical shape [2, 3], physical shape [4, 2, 3]
    let x_data = vec![1.0f32; 4 * 2 * 3];
    let x = FuncTensor::<Flex, 3>::from_data(TensorData::new(x_data, [4, 2, 3]), &device);
    // Batch size N = 4. Logical shape [3], physical shape [4, 3]
    let y_data = vec![
        1.0f32, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0,
    ];
    let y = FuncTensor::<Flex, 2>::from_data(TensorData::new(y_data, [4, 3]), &device);

    let f = |(x, y): &(FuncTensor<Vmap<Flex>, 3>, FuncTensor<Vmap<Flex>, 2>)| {
        x.clone().add_broadcast(y.clone())
    };
    let out: FuncTensor<Flex, 3> = vmap::<Flex, _, _, _>(f)(&(x, y));
    assert_eq!(out.shape().as_slice(), &[4, 2, 3]);
    let vals = out.into_data_sync().try_to_vec::<f32>().unwrap();
    for b in 0..4 {
        for row in 0..2 {
            let offset = b * 6 + row * 3;
            assert_eq!(&vals[offset..offset + 3], &[2.0, 3.0, 4.0]);
        }
    }
}

#[test]
fn vmap_2d_matmul_with_unbatched_weight() {
    let device = FlexDevice;
    // Batch size N = 4. Logical shape [2, 3], physical shape [4, 2, 3]
    let x_data = vec![
        // sample 0
        1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, // sample 1
        0.0, 1.0, 0.0, 0.0, 0.0, 1.0, // sample 2
        1.0, 1.0, 0.0, 0.0, 1.0, 1.0, // sample 3
        2.0, 0.0, 0.0, 0.0, 2.0, 0.0,
    ];
    let x = FuncTensor::<Flex, 3>::from_data(TensorData::new(x_data, [4, 2, 3]), &device);
    // Unbatched weight of shape [3, 2]
    let w = FuncTensor::<Flex, 2>::from_data(
        TensorData::from([[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        &device,
    );

    let f = |(x, w): &(FuncTensor<Vmap<Flex>, 3>, FuncTensor<Vmap<Flex>, 2>)| {
        x.clone().matmul_broadcast(w.clone())
    };
    let out: FuncTensor<Flex, 3> = vmap_with_axes::<Flex, _, _, _>(
        f,
        vec![VmapAxis::Batched(0), VmapAxis::Unbatched],
    )(&(x, w));

    assert_eq!(out.shape().as_slice(), &[4, 2, 2]);
    let vals = out.into_data_sync().try_to_vec::<f32>().unwrap();
    // sample 0: [[1, 0, 0], [0, 1, 0]] @ w = [[1, 2], [3, 4]]
    assert_eq!(&vals[0..4], &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn vmap_relu_on_unbatched_parameter() {
    let device = FlexDevice;
    // Batched x of shape [2, 2] (batch N=2, logical [2])
    let x =
        FuncTensor::<Flex, 2>::from_data(TensorData::from([[1.0f32, 2.0], [3.0, 4.0]]), &device);
    // Unbatched parameter with negative values
    let w = FuncTensor::<Flex, 1>::from_data(TensorData::from([-5.0f32, 10.0]), &device);

    let f = |(x, w): &(FuncTensor<Vmap<Flex>, 2>, FuncTensor<Vmap<Flex>, 1>)| {
        x.clone().add_broadcast(w.clone().relu())
    };
    let out: FuncTensor<Flex, 2> = vmap_with_axes::<Flex, _, _, _>(
        f,
        vec![VmapAxis::Batched(0), VmapAxis::Unbatched],
    )(&(x, w));

    let vals = out.into_data_sync().try_to_vec::<f32>().unwrap();
    assert_eq!(vals, vec![1.0, 12.0, 3.0, 14.0]);
}

#[test]
#[allow(clippy::type_complexity, clippy::identity_op)]
fn vmap_conv2d_avg_pool2d_max_pool2d_attention() {
    use burn_backend::ops::ConvOptions;

    let device = FlexDevice;
    // Physical shape [2, 1, 1, 4, 4] for batch size N=2 of logical 4D inputs [1, 1, 4, 4]
    let x_data = vec![1.0f32; 2 * 1 * 1 * 4 * 4];
    let x = FuncTensor::<Flex, 5>::from_data(TensorData::new(x_data, [2, 1, 1, 4, 4]), &device);

    // 1. conv2d with unbatched weight [1, 1, 3, 3]
    let weight = FuncTensor::<Flex, 4>::ones([1, 1, 3, 3], &device);
    let f_conv = |(x, w): &(FuncTensor<Vmap<Flex>, 5>, FuncTensor<Vmap<Flex>, 4>)| {
        x.clone()
            .conv2d(w.clone(), None, ConvOptions::new([1, 1], [0, 0], [1, 1], 1))
    };
    let out_conv: FuncTensor<Flex, 5> = vmap_with_axes::<Flex, _, _, _>(
        f_conv,
        vec![VmapAxis::Batched(0), VmapAxis::Unbatched],
    )(&(x.clone(), weight));
    assert_eq!(out_conv.shape().as_slice(), &[2, 1, 1, 2, 2]);
    let conv_vals = out_conv.into_data_sync().try_to_vec::<f32>().unwrap();
    assert_eq!(conv_vals, vec![9.0f32; 2 * 1 * 1 * 2 * 2]);

    // 2. avg_pool2d
    let f_avg = |x: &FuncTensor<Vmap<Flex>, 5>| x.clone().avg_pool2d([2, 2], [2, 2], [0, 0], false);
    let out_avg: FuncTensor<Flex, 5> = vmap::<Flex, _, _, _>(f_avg)(&x);
    assert_eq!(out_avg.shape().as_slice(), &[2, 1, 1, 2, 2]);
    let avg_vals = out_avg.into_data_sync().try_to_vec::<f32>().unwrap();
    assert_eq!(avg_vals, vec![1.0f32; 2 * 1 * 1 * 2 * 2]);

    // 3. max_pool2d
    let f_max =
        |x: &FuncTensor<Vmap<Flex>, 5>| x.clone().max_pool2d([2, 2], [2, 2], [0, 0], [1, 1]);
    let out_max: FuncTensor<Flex, 5> = vmap::<Flex, _, _, _>(f_max)(&x);
    assert_eq!(out_max.shape().as_slice(), &[2, 1, 1, 2, 2]);
    let max_vals = out_max.into_data_sync().try_to_vec::<f32>().unwrap();
    assert_eq!(max_vals, vec![1.0f32; 2 * 1 * 1 * 2 * 2]);

    // 4. attention: logical [1, 1, 2, 2], physical [2, 1, 1, 2, 2]
    let q = FuncTensor::<Flex, 5>::from_data(
        TensorData::new(vec![1.0f32; 2 * 1 * 1 * 2 * 2], [2, 1, 1, 2, 2]),
        &device,
    );
    let k = q.clone();
    let v = q.clone();
    let f_attn = |(q, k, v): &(
        FuncTensor<Vmap<Flex>, 5>,
        FuncTensor<Vmap<Flex>, 5>,
        FuncTensor<Vmap<Flex>, 5>,
    )| { q.clone().attention(k.clone(), v.clone()) };
    let out_attn: FuncTensor<Flex, 5> = vmap::<Flex, _, _, _>(f_attn)(&(q, k, v));
    assert_eq!(out_attn.shape().as_slice(), &[2, 1, 1, 2, 2]);
    let attn_vals = out_attn.into_data_sync().try_to_vec::<f32>().unwrap();
    assert!(attn_vals.iter().all(|&v| v.is_finite()));
}

#[test]
#[should_panic(
    expected = "vmap_with_axes: number of axes (1) does not match number of input leaves (2)"
)]
fn vmap_with_axes_length_mismatch_panics() {
    let device = FlexDevice;
    let x = FuncTensor::<Flex, 1>::from_data(TensorData::from([1.0f32, 2.0]), &device);
    let y = FuncTensor::<Flex, 1>::from_data(TensorData::from([3.0f32, 4.0]), &device);
    let f = |(x, y): &(FuncTensor<Vmap<Flex>, 1>, FuncTensor<Vmap<Flex>, 1>)| x.clone() + y.clone();
    let _: FuncTensor<Flex, 1> =
        vmap_with_axes::<Flex, _, _, _>(f, vec![VmapAxis::Batched(0)])(&(x, y));
}
