use burn_autodiff::Autodiff;
use burn_backend::TensorData;
use burn_flex::{Flex, FlexDevice};
use burn_func::{FuncTensor, PyTree, grad};

#[derive(Clone, Debug, PyTree)]
struct Params<B: burn_func::burn_backend::Backend> {
    w: FuncTensor<B, 1>,
    b: FuncTensor<B, 1>,
    #[pytree(skip)]
    metadata: usize,
}

#[test]
fn derive_grad_struct() {
    let device = FlexDevice;
    let params = Params::<Flex> {
        w: FuncTensor::from_data(TensorData::from([2.0f32]), &device),
        b: FuncTensor::from_data(TensorData::from([1.0f32]), &device),
        metadata: 7,
    };

    let f = |p: &Params<Autodiff<Flex>>| {
        let x = FuncTensor::<Autodiff<Flex>, 1>::from_data(TensorData::from([3.0f32]), &FlexDevice);
        (p.w.clone() * x + p.b.clone()).sum()
    };

    let grads = grad::<Flex, _, _>(f)(&params);

    assert_eq!(
        grads.w.into_data_sync().try_to_vec::<f32>().unwrap(),
        vec![3.0]
    );
    assert_eq!(
        grads.b.into_data_sync().try_to_vec::<f32>().unwrap(),
        vec![1.0]
    );
    assert_eq!(grads.metadata, 7);
}

#[test]
fn tuple_tree() {
    let device = FlexDevice;
    let params = (
        FuncTensor::<Flex, 1>::from_data(TensorData::from([2.0f32]), &device),
        FuncTensor::<Flex, 1>::from_data(TensorData::from([5.0f32]), &device),
    );

    let f = |p: &(FuncTensor<Autodiff<Flex>, 1>, FuncTensor<Autodiff<Flex>, 1>)| {
        (p.0.clone() * 2.0f64 + p.1.clone() * 4.0f64).sum()
    };

    let grads = grad::<Flex, _, _>(f)(&params);

    assert_eq!(
        grads.0.into_data_sync().try_to_vec::<f32>().unwrap(),
        vec![2.0]
    );
    assert_eq!(
        grads.1.into_data_sync().try_to_vec::<f32>().unwrap(),
        vec![4.0]
    );
}
