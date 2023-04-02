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
    use super::*;
    use crate::{module::Module, nn, TestBackend};

    #[test]
    fn test_can_save_and_load_bin_format() {
        test_can_save_and_load::<InMemoryBinRecorder>()
    }

    fn test_can_save_and_load<Recorder: InMemoryRecorder>() {
        let model_before = create_model();
        let state_before = model_before.state();
        let bytes = Recorder::record(state_before.clone(), ()).unwrap();

        let model_after = create_model()
            .load(&Recorder::load(bytes).unwrap())
            .unwrap();

        let state_after = model_after.state();
        assert_eq!(state_before, state_after);
    }

    pub fn create_model() -> nn::Linear<TestBackend> {
        nn::LinearConfig::new(32, 32).with_bias(true).init()
    }
}
