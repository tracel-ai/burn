use super::{ElementWise, ExecutionPhase};
use crate::{FloatElement, GraphicsApi, IntElement, Wgpu};
use burn_compute::tune::{AutotuneOperation, AutotuneOperationSet, AutotuneOperationSetStateful};
use burn_fusion::{stream::Context, Fusion};
use burn_tensor::{backend::Backend, Device, Shape, Tensor};

impl<G, F, I> AutotuneOperationSetStateful<String> for ElementWise<G, F, I, ExecutionPhase>
where
    G: GraphicsApi,
    F: FloatElement,
    I: IntElement,
{
    type State<'a> = Context<'a, Wgpu<G, F, I>>;

    fn key(&self) -> String {
        "Allo".to_string()
    }

    fn autotunables(&self) -> Vec<Box<dyn burn_compute::tune::AutotuneOperation>> {
        // TODO: Make sure on another stream to NOT reset the stream of the fastest operation.
        vec![Box::new(FusionAutotune::<G, F, I, 2>::new(
            Default::default(),
            Shape::new([32, 32]),
        ))]
    }

    fn fastest(
        self,
        fastest_index: usize,
        state: &mut Self::State<'_>,
    ) -> Box<dyn AutotuneOperation> {
        let kernel = self.phase.kernel_set_1.select(
            &self.inputs.iter().map(|a| &a.0).collect::<Vec<_>>(),
            &self.outputs.iter().map(|a| &a.0).collect::<Vec<_>>(),
            self.scalars.num_f32,
            self.scalars.num_i32,
            state,
            self.device.clone(),
        );

        Box::new(kernel)
    }
}

#[derive(new, Clone)]
pub struct FusionAutotune<G: GraphicsApi, F: FloatElement, I: IntElement, const D: usize> {
    device: Device<Wgpu<G, F, I>>,
    shape: Shape<D>,
}

impl<G, F, I, const D: usize> AutotuneOperation for FusionAutotune<G, F, I, D>
where
    G: GraphicsApi,
    F: FloatElement,
    I: IntElement,
{
    fn execute(self: Box<Self>) {
        let mut tensor = Tensor::<Fusion<Wgpu<G, F, I>>, 2>::random(
            [32, 32],
            burn_tensor::Distribution::Default,
            &self.device,
        );
        for _i in 0..8 {
            tensor = tensor.log();
        }

        Fusion::<Wgpu<G, F, I>>::sync(&self.device);
    }

    fn clone(&self) -> Box<dyn AutotuneOperation> {
        Box::new(Clone::clone(self))
    }
}
