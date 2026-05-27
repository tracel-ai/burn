use crate::{
    Backend, BackendTypes, ComplexTensorBackend, InterleavedLayout, UnimplementedTensorPrimitive,
    ops::ComplexTensorOps,
};

const fn complex_panic_message() -> &'static str {
    "Interleaved complex tensors are not implemented for this backend. Use split complex tensors instead"
}

impl<B, C> ComplexTensorBackend for B
where
    B: Backend + BackendTypes<ComplexTensorPrimitive = UnimplementedTensorPrimitive<C>>,
    C: Clone + Send + Sync + 'static,
{
    type InnerBackend = Self;

    type Layout = InterleavedLayout;

    fn complex_from_real_data(
        _data: burn_std::TensorData,
        _device: &Self::Device,
    ) -> crate::ComplexTensor<Self> {
        panic!("{}", complex_panic_message())
    }

    fn complex_from_imag_data(
        _data: burn_std::TensorData,
        _device: &Self::Device,
    ) -> crate::ComplexTensor<Self> {
        panic!("{}", complex_panic_message())
    }

    fn complex_from_interleaved_data(
        _data: burn_std::TensorData,
        _device: &Self::Device,
    ) -> crate::ComplexTensor<Self> {
        panic!("{}", complex_panic_message())
    }

    fn complex_from_parts_data(
        _real_data: burn_std::TensorData,
        _imag_data: burn_std::TensorData,
        _device: &Self::Device,
    ) -> crate::ComplexTensor<Self> {
        panic!("{}", complex_panic_message())
    }
}

impl<B, C> ComplexTensorOps<B> for B
where
    B: Backend + BackendTypes<ComplexTensorPrimitive = UnimplementedTensorPrimitive<C>>,
    C: Clone + Send + Sync + 'static,
{
    async fn complex_into_real_data(
        _tensor: crate::ComplexTensor<B>,
    ) -> Result<burn_std::TensorData, burn_std::ExecutionError> {
        panic!("{}", complex_panic_message())
    }

    async fn complex_into_imag_data(
        _tensor: crate::ComplexTensor<B>,
    ) -> Result<burn_std::TensorData, burn_std::ExecutionError> {
        panic!("{}", complex_panic_message())
    }

    async fn complex_into_interleaved_data(
        _tensor: crate::ComplexTensor<B>,
    ) -> Result<burn_std::TensorData, burn_std::ExecutionError> {
        panic!("{}", complex_panic_message())
    }

    async fn complex_into_split_data(
        _tensor: crate::ComplexTensor<B>,
    ) -> Result<burn_std::SplitTensorData, burn_std::ExecutionError> {
        panic!("{}", complex_panic_message())
    }

    // fn to_complex(_tensor: crate::tensor::FloatTensor<B>) -> crate::ComplexTensor<B> {
    //     panic!("{}",complex_panic_message())
    // }

    fn complex_squared_norm(_tensor: crate::ComplexTensor<B>) -> crate::tensor::FloatTensor<B> {
        panic!("{}", complex_panic_message())
    }

    fn complex_random(
        _shape: burn_std::Shape,
        _distribution: burn_std::Distribution,
        _device: &crate::tensor::Device<B>,
        _dtype: burn_std::ComplexDType,
    ) -> crate::ComplexTensor<B> {
        panic!("{}", complex_panic_message())
    }

    fn complex_to_device(
        _tensor: crate::ComplexTensor<B>,
        _device: &<B>::Device,
    ) -> crate::ComplexTensor<B> {
        panic!("{}", complex_panic_message())
    }

    async fn complex_into_data(
        _tensor: crate::ComplexTensor<B>,
    ) -> Result<super::OutTensorData<B>, burn_std::ExecutionError> {
        panic!("{}", complex_panic_message())
    }

    fn complex_reshape(
        _tensor: crate::ComplexTensor<B>,
        _shape: burn_std::Shape,
    ) -> crate::ComplexTensor<B> {
        panic!("{}", complex_panic_message())
    }

    fn complex_transpose(_tensor: crate::ComplexTensor<B>) -> crate::ComplexTensor<B> {
        panic!("{}", complex_panic_message())
    }

    fn complex_add(
        _lhs: crate::ComplexTensor<B>,
        _rhs: crate::ComplexTensor<B>,
    ) -> crate::ComplexTensor<B> {
        panic!("{}", complex_panic_message())
    }

    fn complex_sub(
        _lhs: crate::ComplexTensor<B>,
        _rhs: crate::ComplexTensor<B>,
    ) -> crate::ComplexTensor<B> {
        panic!("{}", complex_panic_message())
    }

    fn complex_mul(
        _lhs: crate::ComplexTensor<B>,
        _rhs: crate::ComplexTensor<B>,
    ) -> crate::ComplexTensor<B> {
        panic!("{}", complex_panic_message())
    }

    fn complex_div(
        _lhs: crate::ComplexTensor<B>,
        _rhs: crate::ComplexTensor<B>,
    ) -> crate::ComplexTensor<B> {
        panic!("{}", complex_panic_message())
    }

    fn complex_neg(_tensor: crate::ComplexTensor<B>) -> crate::ComplexTensor<B> {
        panic!("{}", complex_panic_message())
    }

    fn complex_conj(_tensor: crate::ComplexTensor<B>) -> crate::ComplexTensor<B> {
        panic!("{}", complex_panic_message())
    }

    fn complex_real(_tensor: crate::ComplexTensor<B>) -> crate::tensor::FloatTensor<B> {
        panic!("{}", complex_panic_message())
    }

    fn complex_imag(_tensor: crate::ComplexTensor<B>) -> crate::tensor::FloatTensor<B> {
        panic!("{}", complex_panic_message())
    }

    fn complex_abs(_tensor: crate::ComplexTensor<B>) -> crate::tensor::FloatTensor<B> {
        panic!("{}", complex_panic_message())
    }

    fn complex_arg(_tensor: crate::ComplexTensor<B>) -> crate::tensor::FloatTensor<B> {
        panic!("{}", complex_panic_message())
    }

    fn complex_from_parts(
        _real: burn_std::TensorData,
        _imag: burn_std::TensorData,
        _device: &<B>::Device,
    ) -> crate::ComplexTensor<B> {
        panic!("{}", complex_panic_message())
    }

    fn complex_from_polar(
        _magnitude: crate::tensor::FloatTensor<B>,
        _phase: crate::tensor::FloatTensor<B>,
    ) -> crate::ComplexTensor<B> {
        panic!("{}", complex_panic_message())
    }

    fn complex_exp(_tensor: crate::ComplexTensor<B>) -> crate::ComplexTensor<B> {
        panic!("{}", complex_panic_message())
    }

    fn complex_log(_tensor: crate::ComplexTensor<B>) -> crate::ComplexTensor<B> {
        panic!("{}", complex_panic_message())
    }

    fn complex_powc(
        _lhs: crate::ComplexTensor<B>,
        _rhs: crate::ComplexTensor<B>,
    ) -> crate::ComplexTensor<B> {
        panic!("{}", complex_panic_message())
    }

    fn complex_sqrt(_tensor: crate::ComplexTensor<B>) -> crate::ComplexTensor<B> {
        panic!("{}", complex_panic_message())
    }

    fn complex_sin(_tensor: crate::ComplexTensor<B>) -> crate::ComplexTensor<B> {
        panic!("{}", complex_panic_message())
    }

    fn complex_cos(_tensor: crate::ComplexTensor<B>) -> crate::ComplexTensor<B> {
        panic!("{}", complex_panic_message())
    }

    fn complex_tan(_tensor: crate::ComplexTensor<B>) -> crate::ComplexTensor<B> {
        panic!("{}", complex_panic_message())
    }

    fn complex_acos(_tensor: crate::ComplexTensor<B>) -> crate::ComplexTensor<B> {
        panic!("{}", complex_panic_message())
    }

    fn complex_acosh(_tensor: crate::ComplexTensor<B>) -> crate::ComplexTensor<B> {
        panic!("{}", complex_panic_message())
    }

    fn complex_asin(_tensor: crate::ComplexTensor<B>) -> crate::ComplexTensor<B> {
        panic!("{}", complex_panic_message())
    }

    fn complex_asinh(_tensor: crate::ComplexTensor<B>) -> crate::ComplexTensor<B> {
        panic!("{}", complex_panic_message())
    }

    fn complex_atan(_tensor: crate::ComplexTensor<B>) -> crate::ComplexTensor<B> {
        panic!("{}", complex_panic_message())
    }

    fn complex_atanh(_tensor: crate::ComplexTensor<B>) -> crate::ComplexTensor<B> {
        panic!("{}", complex_panic_message())
    }

    fn complex_select(
        _tensor: crate::ComplexTensor<B>,
        _dim: usize,
        _indices: <B>::IntTensorPrimitive,
    ) -> crate::ComplexTensor<B> {
        panic!("{}", complex_panic_message())
    }

    fn complex_slice(
        _tensor: crate::ComplexTensor<B>,
        _slices: &[burn_std::Slice],
    ) -> crate::ComplexTensor<B> {
        panic!("{}", complex_panic_message())
    }

    fn complex_slice_assign(
        _tensor: crate::ComplexTensor<B>,
        _ranges: &[burn_std::Slice],
        _value: crate::ComplexTensor<B>,
    ) -> crate::ComplexTensor<B> {
        panic!("{}", complex_panic_message())
    }

    fn complex_scatter_nd(
        _tensor: crate::ComplexTensor<B>,
        _indices: <B>::IntTensorPrimitive,
        _value: crate::ComplexTensor<B>,
        _reduction: burn_std::IndexingUpdateOp,
    ) -> crate::ComplexTensor<B> {
        panic!("{}", complex_panic_message())
    }

    fn complex_swap_dims(
        _tensor: crate::ComplexTensor<B>,
        _dim1: usize,
        _dim2: usize,
    ) -> crate::ComplexTensor<B> {
        panic!("{}", complex_panic_message())
    }

    fn complex_repeat_dim(
        _tensor: crate::ComplexTensor<B>,
        _dim: usize,
        _times: usize,
    ) -> crate::ComplexTensor<B> {
        panic!("{}", complex_panic_message())
    }

    fn complex_equal(
        _lhs: crate::ComplexTensor<B>,
        _rhs: crate::ComplexTensor<B>,
        _out_dtype: burn_std::BoolDType,
    ) -> <B>::BoolTensorPrimitive {
        panic!("{}", complex_panic_message())
    }

    fn complex_not_equal(
        _lhs: crate::ComplexTensor<B>,
        _rhs: crate::ComplexTensor<B>,
        _out_dtype: burn_std::BoolDType,
    ) -> <B>::BoolTensorPrimitive {
        panic!("{}", complex_panic_message())
    }

    fn complex_cat(
        _tensors: alloc::vec::Vec<crate::ComplexTensor<B>>,
        _dim: usize,
    ) -> crate::ComplexTensor<B> {
        panic!("{}", complex_panic_message())
    }

    fn complex_any(
        _tensor: crate::ComplexTensor<B>,
        _out_dtype: burn_std::BoolDType,
    ) -> <B>::BoolTensorPrimitive {
        panic!("{}", complex_panic_message())
    }

    fn complex_any_dim(
        _tensor: crate::ComplexTensor<B>,
        _dim: usize,
        _out_dtype: burn_std::BoolDType,
    ) -> <B>::BoolTensorPrimitive {
        panic!("{}", complex_panic_message())
    }

    fn complex_all(
        _tensor: crate::ComplexTensor<B>,
        _out_dtype: burn_std::BoolDType,
    ) -> <B>::BoolTensorPrimitive {
        panic!("{}", complex_panic_message())
    }

    fn complex_all_dim(
        _tensor: crate::ComplexTensor<B>,
        _dim: usize,
        _out_dtype: burn_std::BoolDType,
    ) -> <B>::BoolTensorPrimitive {
        panic!("{}", complex_panic_message())
    }

    fn complex_permute(
        _tensor: crate::ComplexTensor<B>,
        _axes: &[usize],
    ) -> crate::ComplexTensor<B> {
        panic!("{}", complex_panic_message())
    }

    fn complex_expand(
        _tensor: crate::ComplexTensor<B>,
        _shape: burn_std::Shape,
    ) -> crate::ComplexTensor<B> {
        panic!("{}", complex_panic_message())
    }

    fn complex_flip(_tensor: crate::ComplexTensor<B>, _axes: &[usize]) -> crate::ComplexTensor<B> {
        panic!("{}", complex_panic_message())
    }

    fn complex_unfold(
        _tensor: crate::ComplexTensor<B>,
        _dim: usize,
        _size: usize,
        _step: usize,
    ) -> crate::ComplexTensor<B> {
        panic!("{}", complex_panic_message())
    }

    fn complex_select_add(
        _tensor: crate::ComplexTensor<B>,
        _dim: usize,
        _indices: <B>::IntTensorPrimitive,
        _values: crate::ComplexTensor<B>,
    ) -> crate::ComplexTensor<B> {
        panic!("{}", complex_panic_message())
    }

    fn complex_sum(_tensor: crate::ComplexTensor<B>) -> crate::ComplexTensor<B> {
        panic!("{}", complex_panic_message())
    }

    fn complex_sum_dim(_tensor: crate::ComplexTensor<B>, _dim: usize) -> crate::ComplexTensor<B> {
        panic!("{}", complex_panic_message())
    }

    fn complex_prod(_tensor: crate::ComplexTensor<B>) -> crate::ComplexTensor<B> {
        panic!("{}", complex_panic_message())
    }

    fn complex_prod_dim(_tensor: crate::ComplexTensor<B>, _dim: usize) -> crate::ComplexTensor<B> {
        panic!("{}", complex_panic_message())
    }

    fn complex_mean(_tensor: crate::ComplexTensor<B>) -> crate::ComplexTensor<B> {
        panic!("{}", complex_panic_message())
    }

    fn complex_mean_dim(_tensor: crate::ComplexTensor<B>, _dim: usize) -> crate::ComplexTensor<B> {
        panic!("{}", complex_panic_message())
    }

    fn complex_remainder(
        _lhs: crate::ComplexTensor<B>,
        _rhs: crate::ComplexTensor<B>,
    ) -> crate::ComplexTensor<B> {
        panic!("{}", complex_panic_message())
    }

    fn complex_remainder_scalar(
        _lhs: crate::ComplexTensor<B>,
        _rhs: burn_std::Scalar,
    ) -> crate::ComplexTensor<B> {
        panic!("{}", complex_panic_message())
    }

    fn complex_equal_elem(
        _lhs: crate::ComplexTensor<B>,
        _rhs: burn_std::Scalar,
        _out_dtype: burn_std::BoolDType,
    ) -> <B>::BoolTensorPrimitive {
        panic!("{}", complex_panic_message())
    }

    fn complex_not_equal_elem(
        _lhs: crate::ComplexTensor<B>,
        _rhs: burn_std::Scalar,
        _out_dtype: burn_std::BoolDType,
    ) -> <B>::BoolTensorPrimitive {
        panic!("{}", complex_panic_message())
    }

    fn complex_mask_where(
        _tensor: crate::ComplexTensor<B>,
        _mask: <B>::BoolTensorPrimitive,
        _source: crate::ComplexTensor<B>,
    ) -> crate::ComplexTensor<B> {
        panic!("{}", complex_panic_message())
    }

    fn complex_mask_fill(
        _tensor: crate::ComplexTensor<B>,
        _mask: <B>::BoolTensorPrimitive,
        _value: burn_std::Scalar,
    ) -> crate::ComplexTensor<B> {
        panic!("{}", complex_panic_message())
    }

    fn complex_gather(
        _dim: usize,
        _tensor: crate::ComplexTensor<B>,
        _indices: <B>::IntTensorPrimitive,
    ) -> crate::ComplexTensor<B> {
        panic!("{}", complex_panic_message())
    }

    fn complex_scatter_add(
        _dim: usize,
        _tensor: crate::ComplexTensor<B>,
        _indices: <B>::IntTensorPrimitive,
        _values: crate::ComplexTensor<B>,
    ) -> crate::ComplexTensor<B> {
        panic!("{}", complex_panic_message())
    }

    fn complex_sign(_tensor: crate::ComplexTensor<B>) -> crate::ComplexTensor<B> {
        panic!("{}", complex_panic_message())
    }

    fn complex_powc_scalar(
        _lhs: crate::ComplexTensor<B>,
        _rhs: burn_std::Scalar,
    ) -> crate::ComplexTensor<B> {
        panic!("{}", complex_panic_message())
    }

    fn complex_powf(
        _lhs: crate::ComplexTensor<B>,
        _rhs: crate::tensor::FloatTensor<B>,
    ) -> crate::ComplexTensor<B> {
        panic!("{}", complex_panic_message())
    }

    fn complex_powf_scalar(
        _lhs: crate::ComplexTensor<B>,
        _rhs: burn_std::Scalar,
    ) -> crate::ComplexTensor<B> {
        panic!("{}", complex_panic_message())
    }

    fn complex_matmul(
        _lhs: crate::ComplexTensor<B>,
        _rhs: crate::ComplexTensor<B>,
    ) -> crate::ComplexTensor<B> {
        panic!("{}", complex_panic_message())
    }

    fn complex_cumsum(_tensor: crate::ComplexTensor<B>, _dim: usize) -> crate::ComplexTensor<B> {
        panic!("{}", complex_panic_message())
    }

    fn complex_cumprod(_tensor: crate::ComplexTensor<B>, _dim: usize) -> crate::ComplexTensor<B> {
        panic!("{}", complex_panic_message())
    }
}
