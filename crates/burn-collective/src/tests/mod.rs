mod all_reduce;
mod broadcast;
mod reduce;

pub(crate) fn read_tensor<B: burn_backend::Backend>(
    tensor: B::FloatTensorPrimitive,
) -> burn_backend::TensorData {
    burn_backend::read_sync(B::float_into_data(tensor)).unwrap()
}
