use super::{PrecisionSettings, Recorder, RecorderError, bin_config};
use burn_tensor::backend::Backend;
use core::marker::PhantomData;
use flate2::{Compression, read::GzDecoder, write::GzEncoder};
use serde::{Serialize, de::DeserializeOwned};
use std::io::{BufReader, BufWriter};
use std::{fs::File, path::PathBuf};

/// Recorder trait specialized to save and load data to and from files.
pub trait FileRecorder<B: Backend>:
    Recorder<B, RecordArgs = PathBuf, RecordOutput = (), LoadArgs = PathBuf>
{
    /// File extension of the format used by the recorder.
    fn file_extension() -> &'static str;
}

/// Default [file recorder](FileRecorder).
pub type DefaultFileRecorder<S> = NamedMpkFileRecorder<S>;

/// File recorder using the [bincode format](bincode).
#[derive(new, Debug, Default, Clone)]
pub struct BinFileRecorder<S: PrecisionSettings> {
    _settings: PhantomData<S>,
}

/// File recorder using the [bincode format](bincode) compressed with gzip.
#[derive(new, Debug, Default, Clone)]
pub struct BinGzFileRecorder<S: PrecisionSettings> {
    _settings: PhantomData<S>,
}

/// File recorder using the [json format](serde_json) compressed with gzip.
#[derive(new, Debug, Default, Clone)]
pub struct JsonGzFileRecorder<S: PrecisionSettings> {
    _settings: PhantomData<S>,
}

/// File recorder using [pretty json format](serde_json) for easy readability.
#[derive(new, Debug, Default, Clone)]
pub struct PrettyJsonFileRecorder<S: PrecisionSettings> {
    _settings: PhantomData<S>,
}

/// File recorder using the [named msgpack](rmp_serde) format compressed with gzip.
#[derive(new, Debug, Default, Clone)]
pub struct NamedMpkGzFileRecorder<S: PrecisionSettings> {
    _settings: PhantomData<S>,
}

/// File recorder using the [named msgpack](rmp_serde) format.
#[derive(new, Debug, Default, Clone)]
pub struct NamedMpkFileRecorder<S: PrecisionSettings> {
    _settings: PhantomData<S>,
}

impl<S: PrecisionSettings, B: Backend> FileRecorder<B> for BinGzFileRecorder<S> {
    fn file_extension() -> &'static str {
        "bin.gz"
    }
}
impl<S: PrecisionSettings, B: Backend> FileRecorder<B> for BinFileRecorder<S> {
    fn file_extension() -> &'static str {
        "bin"
    }
}
impl<S: PrecisionSettings, B: Backend> FileRecorder<B> for JsonGzFileRecorder<S> {
    fn file_extension() -> &'static str {
        "json.gz"
    }
}
impl<S: PrecisionSettings, B: Backend> FileRecorder<B> for PrettyJsonFileRecorder<S> {
    fn file_extension() -> &'static str {
        "json"
    }
}

impl<S: PrecisionSettings, B: Backend> FileRecorder<B> for NamedMpkGzFileRecorder<S> {
    fn file_extension() -> &'static str {
        "mpk.gz"
    }
}

impl<S: PrecisionSettings, B: Backend> FileRecorder<B> for NamedMpkFileRecorder<S> {
    fn file_extension() -> &'static str {
        "mpk"
    }
}

macro_rules! str2reader {
    (
        $file:expr
    ) => {{
        $file.set_extension(<Self as FileRecorder<B>>::file_extension());
        let path = $file.as_path();

        File::open(path)
            .map_err(|err| match err.kind() {
                std::io::ErrorKind::NotFound => RecorderError::FileNotFound(err.to_string()),
                _ => RecorderError::Unknown(err.to_string()),
            })
            .map(|file| BufReader::new(file))
    }};
}

macro_rules! str2writer {
    (
        $file:expr
    ) => {{
        $file.set_extension(<Self as FileRecorder<B>>::file_extension());
        let path = $file.as_path();

        log::debug!("Writing to file: {:?}", path);

        // Add parent directories if they don't exist
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }

        if path.exists() {
            log::warn!("File exists, replacing");
            std::fs::remove_file(path).map_err(|err| RecorderError::Unknown(err.to_string()))?;
        }

        File::create(path)
            .map_err(|err| match err.kind() {
                std::io::ErrorKind::NotFound => RecorderError::FileNotFound(err.to_string()),
                _ => RecorderError::Unknown(err.to_string()),
            })
            .map(|file| BufWriter::new(file))
    }};
}

impl<S: PrecisionSettings, B: Backend> Recorder<B> for BinGzFileRecorder<S> {
    type Settings = S;
    type RecordArgs = PathBuf;
    type RecordOutput = ();
    type LoadArgs = PathBuf;

    fn save_item<I: Serialize>(
        &self,
        item: I,
        mut file: Self::RecordArgs,
    ) -> Result<(), RecorderError> {
        let config = bin_config();
        let writer = str2writer!(file)?;
        let mut writer = GzEncoder::new(writer, Compression::default());

        bincode::serde::encode_into_std_write(&item, &mut writer, config)
            .map_err(|err| RecorderError::Unknown(err.to_string()))?;

        Ok(())
    }

    fn load_item<I: DeserializeOwned>(
        &self,
        file: &mut Self::LoadArgs,
    ) -> Result<I, RecorderError> {
        let reader = str2reader!(file)?;
        let mut reader = GzDecoder::new(reader);
        let state = bincode::serde::decode_from_std_read(&mut reader, bin_config())
            .map_err(|err| RecorderError::Unknown(err.to_string()))?;

        Ok(state)
    }
}

impl<S: PrecisionSettings, B: Backend> Recorder<B> for BinFileRecorder<S> {
    type Settings = S;
    type RecordArgs = PathBuf;
    type RecordOutput = ();
    type LoadArgs = PathBuf;

    fn save_item<I: Serialize>(
        &self,
        item: I,
        mut file: Self::RecordArgs,
    ) -> Result<(), RecorderError> {
        let config = bin_config();
        let mut writer = str2writer!(file)?;
        bincode::serde::encode_into_std_write(&item, &mut writer, config)
            .map_err(|err| RecorderError::Unknown(err.to_string()))?;
        Ok(())
    }

    fn load_item<I: DeserializeOwned>(
        &self,
        file: &mut Self::LoadArgs,
    ) -> Result<I, RecorderError> {
        let mut reader = str2reader!(file)?;
        let state = bincode::serde::decode_from_std_read(&mut reader, bin_config())
            .map_err(|err| RecorderError::Unknown(err.to_string()))?;
        Ok(state)
    }
}

impl<S: PrecisionSettings, B: Backend> Recorder<B> for JsonGzFileRecorder<S> {
    type Settings = S;
    type RecordArgs = PathBuf;
    type RecordOutput = ();
    type LoadArgs = PathBuf;

    fn save_item<I: Serialize>(
        &self,
        item: I,
        mut file: Self::RecordArgs,
    ) -> Result<(), RecorderError> {
        let writer = str2writer!(file)?;
        let writer = GzEncoder::new(writer, Compression::default());
        serde_json::to_writer(writer, &item)
            .map_err(|err| RecorderError::Unknown(err.to_string()))?;

        Ok(())
    }

    fn load_item<I: DeserializeOwned>(
        &self,
        file: &mut Self::LoadArgs,
    ) -> Result<I, RecorderError> {
        let reader = str2reader!(file)?;
        let reader = GzDecoder::new(reader);
        let state = serde_json::from_reader(reader)
            .map_err(|err| RecorderError::Unknown(err.to_string()))?;

        Ok(state)
    }
}

impl<S: PrecisionSettings, B: Backend> Recorder<B> for PrettyJsonFileRecorder<S> {
    type Settings = S;
    type RecordArgs = PathBuf;
    type RecordOutput = ();
    type LoadArgs = PathBuf;

    fn save_item<I: Serialize>(
        &self,
        item: I,
        mut file: Self::RecordArgs,
    ) -> Result<(), RecorderError> {
        let writer = str2writer!(file)?;
        serde_json::to_writer_pretty(writer, &item)
            .map_err(|err| RecorderError::Unknown(err.to_string()))?;
        Ok(())
    }

    fn load_item<I: DeserializeOwned>(
        &self,
        file: &mut Self::LoadArgs,
    ) -> Result<I, RecorderError> {
        let reader = str2reader!(file)?;
        let state = serde_json::from_reader(reader)
            .map_err(|err| RecorderError::Unknown(err.to_string()))?;

        Ok(state)
    }
}

impl<S: PrecisionSettings, B: Backend> Recorder<B> for NamedMpkGzFileRecorder<S> {
    type Settings = S;
    type RecordArgs = PathBuf;
    type RecordOutput = ();
    type LoadArgs = PathBuf;

    fn save_item<I: Serialize>(
        &self,
        item: I,
        mut file: Self::RecordArgs,
    ) -> Result<(), RecorderError> {
        let writer = str2writer!(file)?;
        let mut writer = GzEncoder::new(writer, Compression::default());
        rmp_serde::encode::write_named(&mut writer, &item)
            .map_err(|err| RecorderError::Unknown(err.to_string()))?;

        Ok(())
    }

    fn load_item<I: DeserializeOwned>(
        &self,
        file: &mut Self::LoadArgs,
    ) -> Result<I, RecorderError> {
        let reader = str2reader!(file)?;
        let reader = GzDecoder::new(reader);
        let state = rmp_serde::decode::from_read(reader)
            .map_err(|err| RecorderError::Unknown(err.to_string()))?;

        Ok(state)
    }
}

impl<S: PrecisionSettings, B: Backend> Recorder<B> for NamedMpkFileRecorder<S> {
    type Settings = S;
    type RecordArgs = PathBuf;
    type RecordOutput = ();
    type LoadArgs = PathBuf;

    fn save_item<I: Serialize>(
        &self,
        item: I,
        mut file: Self::RecordArgs,
    ) -> Result<(), RecorderError> {
        let mut writer = str2writer!(file)?;

        rmp_serde::encode::write_named(&mut writer, &item)
            .map_err(|err| RecorderError::Unknown(err.to_string()))?;

        Ok(())
    }

    fn load_item<I: DeserializeOwned>(
        &self,
        file: &mut Self::LoadArgs,
    ) -> Result<I, RecorderError> {
        let reader = str2reader!(file)?;
        let state = rmp_serde::decode::from_read(reader)
            .map_err(|err| RecorderError::Unknown(err.to_string()))?;

        Ok(state)
    }
}

#[allow(deprecated)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate as burn;
    use crate::config::Config;
    use crate::module::Ignored;
    use crate::test_utils::SimpleLinear;
    use crate::{
        TestBackend,
        module::Module,
        record::{BinBytesRecorder, FullPrecisionSettings},
    };
    use burn_tensor::Tensor;
    use burn_tensor::backend::Backend;

    #[inline(always)]
    fn file_path(file: &str) -> PathBuf {
        std::env::temp_dir().as_path().join(file)
    }

    #[test]
    fn test_can_save_and_load_jsongz_format() {
        test_can_save_and_load(JsonGzFileRecorder::<FullPrecisionSettings>::default())
    }

    #[test]
    fn test_can_save_and_load_bin_format() {
        test_can_save_and_load(BinFileRecorder::<FullPrecisionSettings>::default())
    }

    #[test]
    fn test_can_save_and_load_bingz_format() {
        test_can_save_and_load(BinGzFileRecorder::<FullPrecisionSettings>::default())
    }

    #[test]
    fn test_can_save_and_load_pretty_json_format() {
        test_can_save_and_load(PrettyJsonFileRecorder::<FullPrecisionSettings>::default())
    }

    #[test]
    fn test_can_save_and_load_mpkgz_format() {
        test_can_save_and_load(NamedMpkGzFileRecorder::<FullPrecisionSettings>::default())
    }

    #[test]
    fn test_can_save_and_load_mpk_format() {
        test_can_save_and_load(NamedMpkFileRecorder::<FullPrecisionSettings>::default())
    }

    fn test_can_save_and_load<Recorder>(recorder: Recorder)
    where
        Recorder: FileRecorder<TestBackend>,
    {
        let filename = "burn_test_file_recorder";

        let device = Default::default();
        let mut model_before = create_model(&device);

        // NOTE: Non-module fields currently act like `#[module(skip)]`, meaning their state
        // is not persistent. These fields hold `EmptyRecord`s.
        // So `model_bytes_after == model_bytes_before` because the changes do not persist in the record.
        model_before.tensor = Tensor::full([4], 2., &device);
        model_before.arr = [3, 3];
        model_before.int = 1;
        model_before.ignore = Ignored(PaddingConfig2d::Valid);

        recorder
            .record(model_before.clone().into_record(), file_path(filename))
            .unwrap();

        let model_after =
            create_model(&device).load_record(recorder.load(file_path(filename), &device).unwrap());

        // State is not persisted for empty record fields
        assert_eq!(model_after.arr, [2, 2]);
        assert_eq!(model_after.int, 0);
        assert_eq!(model_after.ignore.0, PaddingConfig2d::Same);

        let byte_recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
        let model_bytes_before = byte_recorder
            .record(model_before.into_record(), ())
            .unwrap();
        let model_bytes_after = byte_recorder.record(model_after.into_record(), ()).unwrap();

        assert_eq!(model_bytes_after, model_bytes_before);
    }

    #[derive(Config, Debug, PartialEq, Eq)]
    pub enum PaddingConfig2d {
        Same,
        Valid,
        Explicit(usize, usize),
    }

    // Dummy model with different record types
    #[derive(Module, Debug)]
    pub struct Model<B: Backend> {
        linear1: SimpleLinear<B>,
        phantom: PhantomData<B>,
        tensor: Tensor<B, 1>,
        arr: [usize; 2],
        int: usize,
        ignore: Ignored<PaddingConfig2d>,
    }

    pub fn create_model(device: &<TestBackend as Backend>::Device) -> Model<TestBackend> {
        let linear1 = SimpleLinear::new(32, 32, device);

        Model {
            linear1,
            phantom: PhantomData,
            tensor: Tensor::zeros([2], device),
            arr: [2, 2],
            int: 0,
            ignore: Ignored(PaddingConfig2d::Same),
        }
    }

    #[derive(Module, Debug)]
    pub struct ModelComposed<B: Backend> {
        linear1: SimpleLinear<B>,
        phantom: PhantomData<B>,
        tensor: Tensor<B, 1>,
        arr: [usize; 2],
        int: usize,
        #[module(skip)] // same as Ignored
        skip: PaddingConfig2d,
        #[module(constant)]
        constant: PaddingConfig2d,
    }

    impl<B: Backend> ModelComposed<B> {
        fn new(device: &B::Device) -> Self {
            Self {
                linear1: SimpleLinear::new(32, 32, device),
                phantom: PhantomData,
                tensor: Tensor::zeros([2], device),
                arr: [2, 2],
                int: 0,
                skip: PaddingConfig2d::Same,
                constant: PaddingConfig2d::Same,
            }
        }
    }

    fn test_can_save_and_load_constants<Recorder>(recorder: Recorder)
    where
        Recorder: FileRecorder<TestBackend>,
    {
        let filename = "burn_test_file_recorder_constants";

        let device = Default::default();
        let mut model_before = ModelComposed::new(&device);

        // NOTE: Only `#[module(constant)]` fields persist
        model_before.tensor = Tensor::full([4], 2., &device);
        model_before.arr = [3, 3];
        model_before.int = 1;
        model_before.skip = PaddingConfig2d::Valid;
        model_before.constant = PaddingConfig2d::Valid;

        recorder
            .record(model_before.clone().into_record(), file_path(filename))
            .unwrap();

        let model_after = ModelComposed::new(&device)
            .load_record(recorder.load(file_path(filename), &device).unwrap());

        // `#[module(skip)]` does not persist, the initial value remains
        assert_eq!(model_after.skip, PaddingConfig2d::Same);
        // `#[module(constant)]` does persist, the saved value is loaded
        assert_eq!(model_after.constant, PaddingConfig2d::Valid);

        let byte_recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
        let model_bytes_before = byte_recorder
            .record(model_before.into_record(), ())
            .unwrap();
        let model_bytes_after = byte_recorder.record(model_after.into_record(), ()).unwrap();

        assert_eq!(model_bytes_after, model_bytes_before);
    }

    #[test]
    fn test_can_save_and_load_constants_mpk_format() {
        test_can_save_and_load_constants(NamedMpkFileRecorder::<FullPrecisionSettings>::default())
    }
}
