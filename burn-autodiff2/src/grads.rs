use burn_tensor::{backend::Backend, container::TensorContainer};

pub type GradID = String;
pub struct Gradients<B: Backend> {
    container: TensorContainer<B, GradID>,
}
