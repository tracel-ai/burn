use super::{bin_config, PrecisionSettings, Recorder, RecorderError};
use alloc::vec::Vec;
use burn_tensor::backend::Backend;
use serde::{de::DeserializeOwned, Serialize};

/// Recorder trait specialized to save and load data to and from bytes.
///
/// # Notes
///
/// This is especially useful in no_std environment where weights are stored directly in
/// compiled binaries.
pub trait BytesRecorder<B: Backend>:
    Recorder<B, RecordArgs = (), RecordOutput = Vec<u8>, LoadArgs = Vec<u8>>
{
}

/// In memory recorder using the [bincode format](bincode).
#[derive(new, Debug, Default, Clone)]
pub struct BinBytesRecorder<S: PrecisionSettings> {
    _settings: core::marker::PhantomData<S>,
}

impl<S: PrecisionSettings, B: Backend> BytesRecorder<B> for BinBytesRecorder<S> {}

impl<S: PrecisionSettings, B: Backend> Recorder<B> for BinBytesRecorder<S> {
    type Settings = S;
    type RecordArgs = ();
    type RecordOutput = Vec<u8>;
    type LoadArgs = Vec<u8>;

    fn save_item<I: Serialize>(
        &self,
        item: I,
        _args: Self::RecordArgs,
    ) -> Result<Self::RecordOutput, RecorderError> {
        Ok(bincode::serde::encode_to_vec(item, bin_config()).unwrap())
    }
    fn load_item<I: DeserializeOwned>(&self, args: Self::LoadArgs) -> Result<I, RecorderError> {
        let state = bincode::serde::decode_borrowed_from_slice(&args, bin_config()).unwrap();
        Ok(state)
    }
}

#[cfg(feature = "std")]
/// In memory recorder using the [Named MessagePack](rmp_serde).
#[derive(new, Debug, Default, Clone)]
pub struct NamedMpkBytesRecorder<S: PrecisionSettings> {
    _settings: core::marker::PhantomData<S>,
}

#[cfg(feature = "std")]
impl<S: PrecisionSettings, B: Backend> BytesRecorder<B> for NamedMpkBytesRecorder<S> {}

#[cfg(feature = "std")]
impl<S: PrecisionSettings, B: Backend> Recorder<B> for NamedMpkBytesRecorder<S> {
    type Settings = S;
    type RecordArgs = ();
    type RecordOutput = Vec<u8>;
    type LoadArgs = Vec<u8>;

    fn save_item<I: Serialize>(
        &self,
        item: I,
        _args: Self::RecordArgs,
    ) -> Result<Self::RecordOutput, RecorderError> {
        rmp_serde::encode::to_vec_named(&item).map_err(|e| RecorderError::Unknown(e.to_string()))
    }
    fn load_item<I: DeserializeOwned>(&self, args: Self::LoadArgs) -> Result<I, RecorderError> {
        rmp_serde::decode::from_slice(&args).map_err(|e| RecorderError::Unknown(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        module::Module, nn, record::FullPrecisionSettings, tensor::backend::Backend, TestBackend,
    };

    #[test]
    fn test_can_save_and_load_bin_format() {
        test_can_save_and_load(BinBytesRecorder::<FullPrecisionSettings>::default())
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_can_save_and_load_named_mpk_format() {
        test_can_save_and_load(NamedMpkBytesRecorder::<FullPrecisionSettings>::default())
    }

    fn test_can_save_and_load<Recorder>(recorder: Recorder)
    where
        Recorder: BytesRecorder<TestBackend>,
    {
        let device = Default::default();
        let model1 = create_model::<TestBackend>(&device);
        let model2 = create_model::<TestBackend>(&device);
        let bytes1 = recorder.record(model1.into_record(), ()).unwrap();
        let bytes2 = recorder.record(model2.clone().into_record(), ()).unwrap();

        let model2_after = model2.load_record(recorder.load(bytes1.clone(), &device).unwrap());
        let bytes2_after = recorder.record(model2_after.into_record(), ()).unwrap();

        assert_ne!(bytes1, bytes2);
        assert_eq!(bytes1, bytes2_after);
    }

    pub fn create_model<B: Backend>(device: &B::Device) -> nn::Linear<B> {
        nn::LinearConfig::new(32, 32).with_bias(true).init(device)
    }
}
