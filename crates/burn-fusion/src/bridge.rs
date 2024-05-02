use crate::{
    client::FusionClient, stream::execution::Operation, Fusion, FusionBackend, FusionRuntime,
};
use burn_tensor::{
    backend::BackendBridge,
    ops::FloatTensor,
    repr::{
        BaseOperationDescription, HandleContainer, OperationDescription, UnaryOperationDescription,
    },
    Element,
};
use std::marker::PhantomData;

#[derive(Debug)]
/// Fusion bridge.
pub struct PrecisionBridge<B: FusionBackend> {
    _b: PhantomData<B>,
}

impl<BInput, BTarget> BackendBridge<Fusion<BInput>> for PrecisionBridge<BTarget>
where
    BInput: FusionBackend,
    BTarget: FusionBackend<FusionRuntime = BInput::FusionRuntime>,
{
    type Target = Fusion<BTarget>;

    fn into_target<const D: usize>(
        tensor: FloatTensor<Fusion<BInput>, D>,
        _device: Option<burn_tensor::Device<Self::Target>>,
    ) -> FloatTensor<Self::Target, D> {
        cast::<BInput, BTarget, D>(tensor)
    }

    fn from_target<const D: usize>(
        tensor: FloatTensor<Self::Target, D>,
        _device: Option<burn_tensor::Device<Fusion<BInput>>>,
    ) -> FloatTensor<Fusion<BInput>, D> {
        cast::<BTarget, BInput, D>(tensor)
    }
}

fn cast<BInput, BTarget, const D: usize>(
    input: FloatTensor<Fusion<BInput>, D>,
) -> FloatTensor<Fusion<BTarget>, D>
where
    BInput: FusionBackend,
    BTarget: FusionBackend<FusionRuntime = BInput::FusionRuntime>,
{
    #[derive(new)]
    struct Cast<BInput: FusionBackend, BTarget: FusionBackend, const D: usize> {
        desc: UnaryOperationDescription,
        _bi: PhantomData<BInput>,
        _bt: PhantomData<BTarget>,
    }

    impl<const D: usize, BInput, BTarget> Operation<BTarget::FusionRuntime> for Cast<BInput, BTarget, D>
    where
        BInput: FusionBackend,
        BTarget: FusionBackend<FusionRuntime = BInput::FusionRuntime>,
    {
        fn execute(
            self: Box<Self>,
            handles: &mut HandleContainer<<BTarget::FusionRuntime as FusionRuntime>::FusionHandle>,
        ) {
            let input = handles.get_float_tensor::<BInput, D>(&self.desc.input);
            let output = BInput::cast_float(input, BTarget::FloatElem::dtype());

            handles.register_handle(self.desc.out.id, output);
        }
    }

    let stream = input.stream;
    let out = input
        .client
        .tensor_uninitialized(input.shape.clone(), BTarget::FloatElem::dtype());

    let desc = UnaryOperationDescription {
        input: input.into_description(),
        out: out.to_description_out(),
    };

    out.client.register(
        vec![stream],
        OperationDescription::BaseFloat(BaseOperationDescription::Cast(desc.clone())),
        Cast::<BInput, BTarget, D>::new(desc),
    );

    out
}
