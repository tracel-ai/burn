use burn_tensor::{ElementConversion, backend::Backend};

use crate::ReduceKind;

pub(crate) fn all_reduce_centralized<B: Backend>(
    tensors: &mut Vec<B::FloatTensorPrimitive>,
    kind: &ReduceKind,
) -> B::FloatTensorPrimitive {
    let tensor_count = tensors.len();
    let mut base = tensors.pop().unwrap();

    for tensor in tensors.drain(..) {
        let target_device = B::float_device(&base);
        let tensor = B::float_to_device(tensor, &target_device);
        base = B::float_add(base, tensor);
    }

    if *kind == ReduceKind::Mean {
        base = B::float_div_scalar(base, (tensor_count as f32).elem());
    }

    base
}
