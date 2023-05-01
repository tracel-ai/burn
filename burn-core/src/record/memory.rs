use core::marker::PhantomData;

use super::{bin_config, BurnRecord, Record, RecordSettings, Recorder, RecorderError};
use alloc::vec::Vec;

/// Recorder trait specialized to save and load data to and from bytes.
///
/// # Notes
///
/// This is especialy useful in no_std environment where weights are stored directly in
/// compiled binaries.
pub trait BytesRecorder:
    Recorder<RecordArgs = (), RecordOutput = Vec<u8>, LoadArgs = Vec<u8>>
{
}

/// In memory recorder using the [bincode format](bincode).
#[derive(new, Debug, Default, Clone)]
pub struct BytesBinRecorder<S: RecordSettings> {
    _settings: PhantomData<S>,
}

impl<S: RecordSettings> BytesRecorder for BytesBinRecorder<S> {}

impl<S: RecordSettings> Recorder for BytesBinRecorder<S> {
    type Settings = S;
    type RecordArgs = ();
    type RecordOutput = Vec<u8>;
    type LoadArgs = Vec<u8>;

    fn save_item<R: Record>(
        &self,
        item: BurnRecord<<R as Record>::Item<S>>,
        _args: Self::RecordArgs,
    ) -> Result<Self::RecordOutput, RecorderError> {
        Ok(bincode::serde::encode_to_vec(item, bin_config()).unwrap())
    }
    fn load_item<R: Record>(
        &self,
        args: Self::LoadArgs,
    ) -> Result<BurnRecord<<R as Record>::Item<S>>, RecorderError> {
        let state = bincode::serde::decode_borrowed_from_slice(&args, bin_config()).unwrap();
        Ok(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{module::Module, nn, record::FullPrecisionSettings, TestBackend};

    #[test]
    fn test_can_save_and_load_bin_format() {
        test_can_save_and_load(BytesBinRecorder::<FullPrecisionSettings>::default())
    }

    fn test_can_save_and_load<Recorder: BytesRecorder>(recorder: Recorder) {
        let model1 = create_model();
        let model2 = create_model();
        let bytes1 = recorder.record(model1.into_record(), ()).unwrap();
        let bytes2 = recorder.record(model2.clone().into_record(), ()).unwrap();

        let model2_after = model2.load_record(recorder.load(bytes1.clone()).unwrap());
        let bytes2_after = recorder.record(model2_after.into_record(), ()).unwrap();

        assert_ne!(bytes1, bytes2);
        assert_eq!(bytes1, bytes2_after);
    }

    pub fn create_model() -> nn::Linear<TestBackend> {
        nn::LinearConfig::new(32, 32).with_bias(true).init()
    }
}
