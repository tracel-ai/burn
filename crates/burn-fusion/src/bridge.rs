use crate::{
    client::FusionClient, stream::execution::Operation, Fusion, FusionBackend, FusionRuntime,
};
use burn_tensor::{
    backend::BackendBridge,
    ops::FloatTensor,
    repr::{
        BaseOperationDescription, CastOperationDescription, HandleContainer, OperationDescription,
    },
    Element,
};
use std::marker::PhantomData;

#[derive(Debug)]
/// Fusion bridge.
pub struct PrecisionBridge<B: FusionBackend> {
    _b: PhantomData<B>,
}

impl<BInput, BOutput> BackendBridge<Fusion<BInput>> for PrecisionBridge<BOutput>
where
    BInput: FusionBackend,
    BOutput: FusionBackend<FusionRuntime = BInput::FusionRuntime, Device = BInput::Device>,
{
    type Target = Fusion<BOutput>;

    fn into_target<const D: usize>(
        tensor: FloatTensor<Fusion<BInput>, D>,
        _device: Option<burn_tensor::Device<Self::Target>>,
    ) -> FloatTensor<Self::Target, D> {
        #[derive(new)]
        struct Cast<BInput: FusionBackend, BTarget: FusionBackend, const D: usize> {
            desc: CastOperationDescription,
            _bi: PhantomData<BInput>,
            _bt: PhantomData<BTarget>,
        }

        impl<const D: usize, BInput, BOutput> Operation<BOutput::FusionRuntime> for Cast<BInput, BOutput, D>
        where
            BInput: FusionBackend,
            BOutput: FusionBackend<FusionRuntime = BInput::FusionRuntime, Device = BInput::Device>,
        {
            fn execute(
                self: Box<Self>,
                handles: &mut HandleContainer<
                    <BOutput::FusionRuntime as FusionRuntime>::FusionHandle,
                >,
            ) {
                let input = handles.get_float_tensor::<BInput, D>(&self.desc.input);
                let output = BInput::cast_float(input, BOutput::FloatElem::dtype());

                handles.register_handle(self.desc.out.id, output);
            }
        }

        let stream = tensor.stream;
        let out = tensor
            .client
            .tensor_uninitialized(tensor.shape.clone(), BOutput::FloatElem::dtype());

        let desc = CastOperationDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };

        out.client.register(
            vec![stream],
            OperationDescription::BaseFloat(BaseOperationDescription::Cast(desc.clone())),
            Cast::<BInput, BOutput, D>::new(desc),
        );

        out.client.clone().to_backend::<BOutput>(out)
    }

    fn from_target<const D: usize>(
        tensor: FloatTensor<Self::Target, D>,
        _device: Option<burn_tensor::Device<Fusion<BInput>>>,
    ) -> FloatTensor<Fusion<BInput>, D> {
        #[derive(new)]
        struct Cast<BInput: FusionBackend, BTarget: FusionBackend, const D: usize> {
            desc: CastOperationDescription,
            _bi: PhantomData<BInput>,
            _bt: PhantomData<BTarget>,
        }

        impl<const D: usize, BInput, BOutput> Operation<BInput::FusionRuntime> for Cast<BInput, BOutput, D>
        where
            BInput: FusionBackend,
            BOutput: FusionBackend<FusionRuntime = BInput::FusionRuntime, Device = BInput::Device>,
        {
            fn execute(
                self: Box<Self>,
                handles: &mut HandleContainer<
                    <BOutput::FusionRuntime as FusionRuntime>::FusionHandle,
                >,
            ) {
                let input = handles.get_float_tensor::<BOutput, D>(&self.desc.input);
                let output = BOutput::cast_float(input, BInput::FloatElem::dtype());

                handles.register_handle(self.desc.out.id, output);
            }
        }

        let stream = tensor.stream;
        let out = tensor
            .client
            .tensor_uninitialized(tensor.shape.clone(), BOutput::FloatElem::dtype());

        let desc = CastOperationDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };

        out.client.register(
            vec![stream],
            OperationDescription::BaseFloat(BaseOperationDescription::Cast(desc.clone())),
            Cast::<BInput, BOutput, D>::new(desc),
        );

        out.client.clone().to_backend::<BInput>(out)
    }
}
