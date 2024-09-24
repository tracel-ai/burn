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
    _backend: PhantomData<B>,
}

impl<R, BInput, BTarget> BackendBridge<Fusion<BInput>> for PrecisionBridge<BTarget>
where
    BInput: FusionBackend<FusionRuntime = R>,
    BTarget: FusionBackend<FusionRuntime = R>,
    R: FusionRuntime + 'static,
{
    type Target = Fusion<BTarget>;

    fn into_target(
        tensor: FloatTensor<Fusion<BInput>>,
        _device: Option<burn_tensor::Device<Self::Target>>,
    ) -> FloatTensor<Self::Target> {
        cast::<R, BInput, BTarget>(tensor)
    }

    fn from_target(
        tensor: FloatTensor<Self::Target>,
        _device: Option<burn_tensor::Device<Fusion<BInput>>>,
    ) -> FloatTensor<Fusion<BInput>> {
        cast::<R, BTarget, BInput>(tensor)
    }
}

fn cast<R, BInput, BTarget>(input: FloatTensor<Fusion<BInput>>) -> FloatTensor<Fusion<BTarget>>
where
    BInput: FusionBackend<FusionRuntime = R>,
    BTarget: FusionBackend<FusionRuntime = R>,
    R: FusionRuntime + 'static,
{
    #[derive(new)]
    struct Cast<R: FusionRuntime, BInput: FusionBackend, BTarget: FusionBackend> {
        desc: UnaryOperationDescription,
        _bi: PhantomData<BInput>,
        _bt: PhantomData<BTarget>,
        _runtime: PhantomData<R>,
    }

    impl<R, BInput, BTarget> Operation<BTarget::FusionRuntime> for Cast<R, BInput, BTarget>
    where
        BInput: FusionBackend<FusionRuntime = R>,
        BTarget: FusionBackend<FusionRuntime = R>,
        R: FusionRuntime,
    {
        fn execute(
            self: Box<Self>,
            handles: &mut HandleContainer<<BTarget::FusionRuntime as FusionRuntime>::FusionHandle>,
        ) {
            let input = handles.get_float_tensor::<BInput>(&self.desc.input);
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
        Cast::<R, BInput, BTarget>::new(desc),
    );

    out
}
