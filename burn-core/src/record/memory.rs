use super::{bin_config, Recorder, RecorderError};
use alloc::vec::Vec;
use serde::{de::DeserializeOwned, Serialize};

/// Recorder trait specialized to save and load data to and from bytes.
///
/// # Notes
///
/// This is especialy useful in no_std environment where weights are stored directly in
/// compiled binaries.
pub trait InMemoryRecorder:
    Recorder<RecordArgs = (), RecordOutput = Vec<u8>, LoadArgs = Vec<u8>>
{
}

/// In memory recorder using the [bincode format](bincode).
#[derive(Debug, Default)]
pub struct InMemoryBinRecorder;

impl InMemoryRecorder for InMemoryBinRecorder {}

impl Recorder for InMemoryBinRecorder {
    type RecordArgs = ();
    type RecordOutput = Vec<u8>;
    type LoadArgs = Vec<u8>;

    fn record<Obj: Serialize + DeserializeOwned>(
        obj: Obj,
        _args: Self::RecordArgs,
    ) -> Result<Vec<u8>, RecorderError> {
        Ok(bincode::serde::encode_to_vec(obj, bin_config()).unwrap())
    }

    fn load<Obj: Serialize + DeserializeOwned>(args: Self::LoadArgs) -> Result<Obj, RecorderError> {
        let state = bincode::serde::decode_borrowed_from_slice(&args, bin_config()).unwrap();
        Ok(state)
    }
}

#[cfg(test)]
mod tests {
    use core::marker::PhantomData;

    use super::*;
    use crate::{
        module::Module,
        nn,
        record::{Record, RecordSettings},
        TestBackend,
    };

    #[test]
    fn test_can_save_and_load_bin_format() {
        test_can_save_and_load::<InMemoryBinRecorder>()
    }

    fn test_can_save_and_load<Recorder: InMemoryRecorder>() {
        #[derive(Debug, Default)]
        struct TestRecordSettings<R> {
            phantom: PhantomData<R>,
        }

        impl<R: crate::record::Recorder> RecordSettings for TestRecordSettings<R> {
            type FloatElem = f32;
            type IntElem = i32;
            type Recorder = R;
        }

        let model1 = create_model();
        let model2 = create_model();
        let bytes1 = model1
            .into_record()
            .record::<TestRecordSettings<Recorder>>(())
            .unwrap();
        let bytes2 = model2
            .clone()
            .into_record()
            .record::<TestRecordSettings<Recorder>>(())
            .unwrap();

        let model2_after = model2.load_record(
            nn::LinearRecord::load::<TestRecordSettings<Recorder>>(bytes1.clone()).unwrap(),
        );
        let bytes2_after = model2_after
            .into_record()
            .record::<TestRecordSettings<Recorder>>(())
            .unwrap();

        assert_ne!(bytes1, bytes2);
        assert_eq!(bytes1, bytes2_after);
    }

    pub fn create_model() -> nn::Linear<TestBackend> {
        nn::LinearConfig::new(32, 32).with_bias(true).init()
    }
}
