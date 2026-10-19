use burn_backend::TensorData;
use burn_flex::{Flex, FlexDevice};
use burn_func::{FuncTensor, Vmap, VmapAxis, vmap, vmap_with_axes};

#[test]
fn vmap_axis_zero() {
    let device = FlexDevice;
    let x =
        FuncTensor::<Flex, 2>::from_data(TensorData::from([[1.0f32, 2.0], [3.0, 4.0]]), &device);

    let f = |x: &FuncTensor<Vmap<Flex>, 2>| (x.clone() * 3.0f64).sum();
    let out: FuncTensor<Flex, 1> = vmap::<Flex, _, _, _>(f)(&x);

    assert_eq!(
        out.into_data_sync().try_to_vec::<f32>().unwrap(),
        vec![9.0, 21.0]
    );
}

#[test]
fn vmap_nonzero_axis() {
    let device = FlexDevice;
    let x =
        FuncTensor::<Flex, 2>::from_data(TensorData::from([[1.0f32, 2.0], [3.0, 4.0]]), &device);

    let f = |x: &FuncTensor<Vmap<Flex>, 2>| (x.clone() * 2.0f64).sum();
    let out: FuncTensor<Flex, 1> =
        vmap_with_axes::<Flex, _, _, _>(f, vec![VmapAxis::Batched(1)])(&x);

    assert_eq!(
        out.into_data_sync().try_to_vec::<f32>().unwrap(),
        vec![8.0, 12.0]
    );
}
