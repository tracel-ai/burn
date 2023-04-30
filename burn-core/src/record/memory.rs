use super::{bin_config, BurnRecord, Record, RecordSettings, Recorder, RecorderError};
use alloc::vec::Vec;

/// Recorder trait specialized to save and load data to and from bytes.
///
/// # Notes
///
/// This is especialy useful in no_std environment where weights are stored directly in
/// compiled binaries.
pub trait BytesRecorder<R: Record, S: RecordSettings>:
    Recorder<R, S, RecordArgs = (), RecordOutput = Vec<u8>, LoadArgs = Vec<u8>>
{
}

/// In memory recorder using the [bincode format](bincode).
#[derive(Debug, Default)]
pub struct BytesBinRecorder;

impl<R: Record, S: RecordSettings> BytesRecorder<R, S> for BytesBinRecorder {}

impl<R: Record, S: RecordSettings> Recorder<R, S> for BytesBinRecorder {
    type RecordArgs = ();
    type RecordOutput = Vec<u8>;
    type LoadArgs = Vec<u8>;

    fn save_item(
        &self,
        item: BurnRecord<<R as Record>::Item<S>>,
        _args: Self::RecordArgs,
    ) -> Result<Self::RecordOutput, RecorderError> {
        Ok(bincode::serde::encode_to_vec(item, bin_config()).unwrap())
    }
    fn load_item(
        &self,
        args: Self::LoadArgs,
    ) -> Result<BurnRecord<<R as Record>::Item<S>>, RecorderError> {
        let state = bincode::serde::decode_borrowed_from_slice(&args, bin_config()).unwrap();
        Ok(state)
    }
}

impl BytesBinRecorder {
    pub fn into_bytes<R: Record, S: RecordSettings>(record: R) -> Result<Vec<u8>, RecorderError> {
        Recorder::<R, S>::record(&BytesBinRecorder, record, ())
    }
    pub fn from_bytes<R: Record, S: RecordSettings>(bytes: Vec<u8>) -> Result<R, RecorderError> {
        Recorder::<R, S>::load(&BytesBinRecorder, bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{module::Module, nn, record::DefaultRecordSettings, TestBackend};

    #[test]
    fn test_can_save_and_load_bin_format() {
        test_can_save_and_load(BytesBinRecorder)
    }

    fn test_can_save_and_load<
        Recorder: BytesRecorder<nn::LinearRecord<TestBackend>, DefaultRecordSettings>,
    >(
        recorder: Recorder,
    ) {
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
