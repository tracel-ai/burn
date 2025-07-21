use burn_tensor::backend::Backend;

pub(crate) fn all_reduce_sum_centralized<B: Backend>(
    tensors: &mut Vec<B::FloatTensorPrimitive>,
) -> B::FloatTensorPrimitive {
    let mut base = tensors.pop().unwrap();

    for tensor in tensors.drain(..) {
        let target_device = B::float_device(&base);
        let tensor = B::float_to_device(tensor, &target_device);
        base = B::float_add(base, tensor);
    }

    base
}
