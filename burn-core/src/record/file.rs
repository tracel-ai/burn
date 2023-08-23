use super::{bin_config, PrecisionSettings, Recorder, RecorderError};
use core::marker::PhantomData;
use flate2::{read::GzDecoder, write::GzEncoder, Compression};
use serde::{de::DeserializeOwned, Serialize};
use std::io::{BufReader, BufWriter};
use std::{fs::File, path::PathBuf};

/// Recorder trait specialized to save and load data to and from files.
pub trait FileRecorder:
    Recorder<RecordArgs = PathBuf, RecordOutput = (), LoadArgs = PathBuf>
{
    /// File extension of the format used by the recorder.
    fn file_extension() -> &'static str;
}

/// Default [file recorder](FileRecorder).
pub type DefaultFileRecorder<S> = NamedMpkGzFileRecorder<S>;

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

impl<S: PrecisionSettings> FileRecorder for BinGzFileRecorder<S> {
    fn file_extension() -> &'static str {
        "bin.gz"
    }
}
impl<S: PrecisionSettings> FileRecorder for BinFileRecorder<S> {
    fn file_extension() -> &'static str {
        "bin"
    }
}
impl<S: PrecisionSettings> FileRecorder for JsonGzFileRecorder<S> {
    fn file_extension() -> &'static str {
        "json.gz"
    }
}
impl<S: PrecisionSettings> FileRecorder for PrettyJsonFileRecorder<S> {
    fn file_extension() -> &'static str {
        "json"
    }
}

impl<S: PrecisionSettings> FileRecorder for NamedMpkGzFileRecorder<S> {
    fn file_extension() -> &'static str {
        "mpk.gz"
    }
}

impl<S: PrecisionSettings> FileRecorder for NamedMpkFileRecorder<S> {
    fn file_extension() -> &'static str {
        "mpk"
    }
}

macro_rules! str2reader {
    (
        $file:expr
    ) => {{
        $file.set_extension(<Self as FileRecorder>::file_extension());
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
        $file.set_extension(<Self as FileRecorder>::file_extension());
        let path = $file.as_path();

        if path.exists() {
            log::info!("File exists, replacing");
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

impl<S: PrecisionSettings> Recorder for BinGzFileRecorder<S> {
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

    fn load_item<I: DeserializeOwned>(&self, mut file: Self::LoadArgs) -> Result<I, RecorderError> {
        let reader = str2reader!(file)?;
        let mut reader = GzDecoder::new(reader);
        let state = bincode::serde::decode_from_std_read(&mut reader, bin_config())
            .map_err(|err| RecorderError::Unknown(err.to_string()))?;

        Ok(state)
    }
}

impl<S: PrecisionSettings> Recorder for BinFileRecorder<S> {
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

    fn load_item<I: DeserializeOwned>(&self, mut file: Self::LoadArgs) -> Result<I, RecorderError> {
        let mut reader = str2reader!(file)?;
        let state = bincode::serde::decode_from_std_read(&mut reader, bin_config())
            .map_err(|err| RecorderError::Unknown(err.to_string()))?;
        Ok(state)
    }
}

impl<S: PrecisionSettings> Recorder for JsonGzFileRecorder<S> {
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

    fn load_item<I: DeserializeOwned>(&self, mut file: Self::LoadArgs) -> Result<I, RecorderError> {
        let reader = str2reader!(file)?;
        let reader = GzDecoder::new(reader);
        let state = serde_json::from_reader(reader)
            .map_err(|err| RecorderError::Unknown(err.to_string()))?;

        Ok(state)
    }
}

impl<S: PrecisionSettings> Recorder for PrettyJsonFileRecorder<S> {
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

    fn load_item<I: DeserializeOwned>(&self, mut file: Self::LoadArgs) -> Result<I, RecorderError> {
        let reader = str2reader!(file)?;
        let state = serde_json::from_reader(reader)
            .map_err(|err| RecorderError::Unknown(err.to_string()))?;

        Ok(state)
    }
}

impl<S: PrecisionSettings> Recorder for NamedMpkGzFileRecorder<S> {
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

    fn load_item<I: DeserializeOwned>(&self, mut file: Self::LoadArgs) -> Result<I, RecorderError> {
        let reader = str2reader!(file)?;
        let reader = GzDecoder::new(reader);
        let state = rmp_serde::decode::from_read(reader)
            .map_err(|err| RecorderError::Unknown(err.to_string()))?;

        Ok(state)
    }
}

impl<S: PrecisionSettings> Recorder for NamedMpkFileRecorder<S> {
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

    fn load_item<I: DeserializeOwned>(&self, mut file: Self::LoadArgs) -> Result<I, RecorderError> {
        let reader = str2reader!(file)?;
        let state = rmp_serde::decode::from_read(reader)
            .map_err(|err| RecorderError::Unknown(err.to_string()))?;

        Ok(state)
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::{
        module::Module,
        nn,
        record::{BinBytesRecorder, FullPrecisionSettings},
        TestBackend,
    };

    static FILE_PATH: &str = "/tmp/burn_test_file_recorder";

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

    fn test_can_save_and_load<Recorder: FileRecorder>(recorder: Recorder) {
        let model_before = create_model();
        recorder
            .record(model_before.clone().into_record(), FILE_PATH.into())
            .unwrap();

        let model_after = create_model().load_record(recorder.load(FILE_PATH.into()).unwrap());

        let byte_recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
        let model_bytes_before = byte_recorder
            .record(model_before.into_record(), ())
            .unwrap();
        let model_bytes_after = byte_recorder.record(model_after.into_record(), ()).unwrap();

        assert_eq!(model_bytes_after, model_bytes_before);
    }

    pub fn create_model() -> nn::Linear<TestBackend> {
        nn::LinearConfig::new(32, 32).with_bias(true).init()
    }
}
