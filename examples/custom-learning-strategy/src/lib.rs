use std::marker::PhantomData;

use burn::tensor::{Device, backend::AutodiffBackend};

pub fn run<B: AutodiffBackend>(device: B::Device) {}

pub struct SingleDeviceLearningStrategy<LC: LearnerComponentTypes> {
    device: Device<LC::Backend>,
    _p: PhantomData<LC>,
}
impl<LC: LearnerComponentTypes> SingleDeviceLearningStrategy<LC> {
    pub fn new(device: Device<LC::Backend>) -> Self {
        Self {
            device,
            _p: PhantomData,
        }
    }
}
