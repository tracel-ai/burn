// impl<B: FusionBackend> ComplexTensorOps<Self> for Fusion<B> {
//     fn complex_from_data(
//         _data: TensorData,
//         _device: &Device<Self>,
//     ) -> FusionTensor<B::FusionRuntime> {
//         unimplemented!("Complex tensor operations are not yet implemented for Fusion backend")
//     }

//     fn complex_random(
//         _shape: Shape,
//         _distribution: Distribution,
//         _device: &Device<Self>,
//     ) -> FusionTensor<B::FusionRuntime> {
//         unimplemented!("Complex tensor operations are not yet implemented for Fusion backend")
//     }

//     fn complex_full(
//         _shape: Shape,
//         _fill_value: B::ComplexElem,
//         _device: &Device<Self>,
//     ) -> FusionTensor<B::FusionRuntime> {
//         unimplemented!("Complex tensor operations are not yet implemented for Fusion backend")
//     }

//     fn complex_shape(_tensor: &FusionTensor<B::FusionRuntime>) -> Shape {
//         unimplemented!("Complex tensor operations are not yet implemented for Fusion backend")
//     }

//     fn complex_to_data(_tensor: &FusionTensor<B::FusionRuntime>) -> TensorData {
//         unimplemented!("Complex tensor operations are not yet implemented for Fusion backend")
//     }

//     fn complex_device(_tensor: &FusionTensor<B::FusionRuntime>) -> Device<Self> {
//         unimplemented!("Complex tensor operations are not yet implemented for Fusion backend")
//     }

//     fn complex_to_device(
//         _tensor: FusionTensor<B::FusionRuntime>,
//         _device: &Device<Self>,
//     ) -> FusionTensor<B::FusionRuntime> {
//         unimplemented!("Complex tensor operations are not yet implemented for Fusion backend")
//     }

//     fn complex_into_data(_tensor: FusionTensor<B::FusionRuntime>) -> TensorData {
//         unimplemented!("Complex tensor operations are not yet implemented for Fusion backend")
//     }

//     fn complex_reshape(
//         _tensor: FusionTensor<B::FusionRuntime>,
//         _shape: Shape,
//     ) -> FusionTensor<B::FusionRuntime> {
//         unimplemented!("Complex tensor operations are not yet implemented for Fusion backend")
//     }

//     fn complex_transpose(
//         _tensor: FusionTensor<B::FusionRuntime>,
//     ) -> FusionTensor<B::FusionRuntime> {
//         unimplemented!("Complex tensor operations are not yet implemented for Fusion backend")
//     }

//     fn complex_add(
//         _lhs: FusionTensor<B::FusionRuntime>,
//         _rhs: FusionTensor<B::FusionRuntime>,
//     ) -> FusionTensor<B::FusionRuntime> {
//         unimplemented!("Complex tensor operations are not yet implemented for Fusion backend")
//     }

//     fn complex_sub(
//         _lhs: FusionTensor<B::FusionRuntime>,
//         _rhs: FusionTensor<B::FusionRuntime>,
//     ) -> FusionTensor<B::FusionRuntime> {
//         unimplemented!("Complex tensor operations are not yet implemented for Fusion backend")
//     }

//     fn complex_mul(
//         _lhs: FusionTensor<B::FusionRuntime>,
//         _rhs: FusionTensor<B::FusionRuntime>,
//     ) -> FusionTensor<B::FusionRuntime> {
//         unimplemented!("Complex tensor operations are not yet implemented for Fusion backend")
//     }

//     fn complex_div(
//         _lhs: FusionTensor<B::FusionRuntime>,
//         _rhs: FusionTensor<B::FusionRuntime>,
//     ) -> FusionTensor<B::FusionRuntime> {
//         unimplemented!("Complex tensor operations are not yet implemented for Fusion backend")
//     }

//     fn complex_neg(_tensor: FusionTensor<B::FusionRuntime>) -> FusionTensor<B::FusionRuntime> {
//         unimplemented!("Complex tensor operations are not yet implemented for Fusion backend")
//     }

//     fn complex_conj(_tensor: FusionTensor<B::FusionRuntime>) -> FusionTensor<B::FusionRuntime> {
//         unimplemented!("Complex tensor operations are not yet implemented for Fusion backend")
//     }

//     fn complex_real(_tensor: FusionTensor<B::FusionRuntime>) -> FusionTensor<B::FusionRuntime> {
//         unimplemented!("Complex tensor operations are not yet implemented for Fusion backend")
//     }

//     fn complex_imag(_tensor: FusionTensor<B::FusionRuntime>) -> FusionTensor<B::FusionRuntime> {
//         unimplemented!("Complex tensor operations are not yet implemented for Fusion backend")
//     }

//     fn complex_abs(_tensor: FusionTensor<B::FusionRuntime>) -> FusionTensor<B::FusionRuntime> {
//         unimplemented!("Complex tensor operations are not yet implemented for Fusion backend")
//     }

//     fn complex_arg(_tensor: FusionTensor<B::FusionRuntime>) -> FusionTensor<B::FusionRuntime> {
//         unimplemented!("Complex tensor operations are not yet implemented for Fusion backend")
//     }

//     fn complex_from_parts(
//         _real: FusionTensor<B::FusionRuntime>,
//         _imag: FusionTensor<B::FusionRuntime>,
//     ) -> FusionTensor<B::FusionRuntime> {
//         unimplemented!("Complex tensor operations are not yet implemented for Fusion backend")
//     }

//     fn complex_from_polar(
//         _magnitude: FusionTensor<B::FusionRuntime>,
//         _phase: FusionTensor<B::FusionRuntime>,
//     ) -> FusionTensor<B::FusionRuntime> {
//         unimplemented!("Complex tensor operations are not yet implemented for Fusion backend")
//     }

//     fn complex_exp(_tensor: FusionTensor<B::FusionRuntime>) -> FusionTensor<B::FusionRuntime> {
//         unimplemented!("Complex tensor operations are not yet implemented for Fusion backend")
//     }

//     fn complex_log(_tensor: FusionTensor<B::FusionRuntime>) -> FusionTensor<B::FusionRuntime> {
//         unimplemented!("Complex tensor operations are not yet implemented for Fusion backend")
//     }

//     fn complex_powc(
//         _lhs: FusionTensor<B::FusionRuntime>,
//         _rhs: FusionTensor<B::FusionRuntime>,
//     ) -> FusionTensor<B::FusionRuntime> {
//         unimplemented!("Complex tensor operations are not yet implemented for Fusion backend")
//     }

//     fn complex_sqrt(_tensor: FusionTensor<B::FusionRuntime>) -> FusionTensor<B::FusionRuntime> {
//         unimplemented!("Complex tensor operations are not yet implemented for Fusion backend")
//     }

//     fn complex_sin(_tensor: FusionTensor<B::FusionRuntime>) -> FusionTensor<B::FusionRuntime> {
//         unimplemented!("Complex tensor operations are not yet implemented for Fusion backend")
//     }

//     fn complex_cos(_tensor: FusionTensor<B::FusionRuntime>) -> FusionTensor<B::FusionRuntime> {
//         unimplemented!("Complex tensor operations are not yet implemented for Fusion backend")
//     }

//     fn complex_tan(_tensor: FusionTensor<B::FusionRuntime>) -> FusionTensor<B::FusionRuntime> {
//         unimplemented!("Complex tensor operations are not yet implemented for Fusion backend")
//     }
// }
