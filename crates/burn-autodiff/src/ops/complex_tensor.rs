// impl<B: Backend, C: CheckpointStrategy> ComplexTensorOps<Self> for Autodiff<B, C> {
//     fn complex_from_data(data: TensorData, device: &Device<Self>) -> ComplexTensor<Self> {
//         B::complex_from_data(data, device)
//     }

//     fn complex_random(
//         shape: Shape,
//         distribution: Distribution,
//         device: &Device<Self>,
//     ) -> ComplexTensor<Self> {
//         B::complex_random(shape, distribution, device)
//     }

//     fn complex_shape(tensor: &ComplexTensor<Self>) -> Shape {
//         B::complex_shape(tensor)
//     }

//     fn complex_to_data(tensor: &ComplexTensor<Self>) -> TensorData {
//         B::complex_to_data(tensor)
//     }

//     fn complex_device(tensor: &ComplexTensor<Self>) -> Device<Self> {
//         B::complex_device(tensor)
//     }

//     fn complex_to_device(
//         tensor: ComplexTensor<Self>,
//         device: &Device<Self>,
//     ) -> ComplexTensor<Self> {
//         B::complex_to_device(tensor, device)
//     }

//     fn complex_into_data(tensor: ComplexTensor<Self>) -> TensorData {
//         B::complex_into_data(tensor)
//     }

//     fn complex_reshape(tensor: ComplexTensor<Self>, shape: Shape) -> ComplexTensor<Self> {
//         B::complex_reshape(tensor, shape)
//     }

//     fn complex_transpose(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
//         B::complex_transpose(tensor)
//     }

//     fn complex_add(lhs: ComplexTensor<Self>, rhs: ComplexTensor<Self>) -> ComplexTensor<Self> {
//         B::complex_add(lhs, rhs)
//     }

//     fn complex_sub(lhs: ComplexTensor<Self>, rhs: ComplexTensor<Self>) -> ComplexTensor<Self> {
//         B::complex_sub(lhs, rhs)
//     }

//     fn complex_mul(lhs: ComplexTensor<Self>, rhs: ComplexTensor<Self>) -> ComplexTensor<Self> {
//         B::complex_mul(lhs, rhs)
//     }

//     fn complex_div(lhs: ComplexTensor<Self>, rhs: ComplexTensor<Self>) -> ComplexTensor<Self> {
//         B::complex_div(lhs, rhs)
//     }

//     fn complex_neg(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
//         B::complex_neg(tensor)
//     }

//     fn complex_conj(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
//         B::complex_conj(tensor)
//     }

//     fn complex_real(_tensor: ComplexTensor<Self>) -> <Self as Backend>::FloatTensorPrimitive {
//         // Since autodiff complex tensors are just the inner backend's complex tensors,
//         // and complex_real returns a float tensor, we need to convert it to an autodiff float tensor
//         todo!("Need to implement autodiff wrapper for complex_real")
//     }

//     fn complex_imag(_tensor: ComplexTensor<Self>) -> <Self as Backend>::FloatTensorPrimitive {
//         todo!("Need to implement autodiff wrapper for complex_imag")
//     }

//     fn complex_abs(_tensor: ComplexTensor<Self>) -> <Self as Backend>::FloatTensorPrimitive {
//         todo!("Need to implement autodiff wrapper for complex_abs")
//     }

//     fn complex_arg(_tensor: ComplexTensor<Self>) -> <Self as Backend>::FloatTensorPrimitive {
//         todo!("Need to implement autodiff wrapper for complex_arg")
//     }

//     fn complex_from_parts(
//         _real: <Self as Backend>::FloatTensorPrimitive,
//         _imag: <Self as Backend>::FloatTensorPrimitive,
//     ) -> ComplexTensor<Self> {
//         todo!("Need to implement autodiff wrapper for complex_from_parts")
//     }

//     fn complex_from_polar(
//         _magnitude: <Self as Backend>::FloatTensorPrimitive,
//         _phase: <Self as Backend>::FloatTensorPrimitive,
//     ) -> ComplexTensor<Self> {
//         todo!("Need to implement autodiff wrapper for complex_from_polar")
//     }

//     fn complex_exp(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
//         B::complex_exp(tensor)
//     }

//     fn complex_log(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
//         B::complex_log(tensor)
//     }

//     fn complex_powc(lhs: ComplexTensor<Self>, rhs: ComplexTensor<Self>) -> ComplexTensor<Self> {
//         B::complex_powc(lhs, rhs)
//     }

//     fn complex_sqrt(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
//         B::complex_sqrt(tensor)
//     }

//     fn complex_sin(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
//         B::complex_sin(tensor)
//     }

//     fn complex_cos(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
//         B::complex_cos(tensor)
//     }

//     fn complex_tan(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
//         B::complex_tan(tensor)
//     }
// }
