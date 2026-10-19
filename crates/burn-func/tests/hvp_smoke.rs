use burn_backend::{Backend, TensorData};
use burn_flex::{Flex, FlexDevice};
use burn_func::{Differentiable, FuncTensor, hvp};

#[derive(Clone)]
struct Cubic;

impl<B: Backend> Differentiable<B, FuncTensor<B, 1>> for Cubic {
    fn eval<B2: Backend>(&self, x: &FuncTensor<B2, 1>) -> FuncTensor<B2, 1> {
        (x.clone() * x.clone() * x.clone()).sum()
    }
}

#[test]
fn hvp_of_cubic() {
    let device = FlexDevice;
    let x = FuncTensor::<Flex, 1>::from_data(TensorData::from([2.0f32]), &device);
    let v = FuncTensor::<Flex, 1>::from_data(TensorData::from([1.0f32]), &device);

    let (gradient, hessian_vector) = hvp::<Flex, _, _>(Cubic, &x, &v);

    assert_eq!(
        gradient.into_data_sync().try_to_vec::<f32>().unwrap(),
        vec![12.0]
    );
    assert_eq!(
        hessian_vector.into_data_sync().try_to_vec::<f32>().unwrap(),
        vec![12.0]
    );
}
