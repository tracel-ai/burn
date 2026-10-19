use burn_autodiff::Autodiff;
use burn_backend::TensorData;
use burn_flex::{Flex, FlexDevice};
use burn_func::{FuncTensor, grad};

type B = Flex;
type AB = Autodiff<B>;

#[test]
fn grad_linear_function() {
    let device = FlexDevice;
    let x = FuncTensor::<B, 1>::from_data(TensorData::from([2.0f32]), &device);

    let f = |x: &FuncTensor<AB, 1>| (x.clone() * 3.0f64).sum();
    let dx = grad::<B, _, _>(f)(&x);

    let values = dx.into_data_sync().try_to_vec::<f32>().unwrap();
    assert_eq!(values, vec![3.0]);
}

#[test]
fn grad_constant_function_is_zero() {
    let device = FlexDevice;
    let x = FuncTensor::<B, 1>::from_data(TensorData::from([2.0f32, 4.0f32]), &device);

    let f = |x: &FuncTensor<AB, 1>| {
        let _ = x;
        FuncTensor::<AB, 1>::ones([1], &FlexDevice)
    };
    let dx = grad::<B, _, _>(f)(&x);

    let values = dx.into_data_sync().try_to_vec::<f32>().unwrap();
    assert_eq!(values, vec![0.0, 0.0]);
}
