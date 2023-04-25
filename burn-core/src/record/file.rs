use super::{bin_config, Recorder, RecorderError};
use flate2::{read::GzDecoder, write::GzEncoder, Compression};
use serde::{de::DeserializeOwned, Serialize};
use std::{fs::File, path::PathBuf};

/// Recorder trait specialized to save and load data to and from files.
pub trait FileRecorder:
    Recorder<RecordArgs = PathBuf, RecordOutput = (), LoadArgs = PathBuf>
{
    fn file_extension() -> &'static str;
}

/// File recorder using the [bincode format](bincode).
#[derive(Debug, Default)]
pub struct FileBinRecorder;

/// File recorder using the [bincode format](bincode) compressed with gzip.
#[derive(Debug, Default)]
pub struct FileBinGzRecorder;

/// File recorder using the json format compressed with gzip.
#[derive(Debug, Default)]
pub struct FileJsonGzRecorder;

/// File recorder using pretty json for easy redability.
#[derive(Debug, Default)]
pub struct FilePrettyJsonRecorder;

/// File recorder using the [named msgpack](rmp_serde) format compressed with gzip.
#[derive(Debug, Default)]
pub struct FileNamedMpkGzRecorder;

impl FileRecorder for FileBinGzRecorder {
    fn file_extension() -> &'static str {
        "bin.gz"
    }
}
impl FileRecorder for FileBinRecorder {
    fn file_extension() -> &'static str {
        "bin"
    }
}
impl FileRecorder for FileJsonGzRecorder {
    fn file_extension() -> &'static str {
        "json.gz"
    }
}
impl FileRecorder for FilePrettyJsonRecorder {
    fn file_extension() -> &'static str {
        "json"
    }
}

impl FileRecorder for FileNamedMpkGzRecorder {
    fn file_extension() -> &'static str {
        "mpk.gz"
    }
}

macro_rules! str2reader {
    (
        $file:expr,
        $ext:expr
    ) => {{
        $file.set_extension($ext);
        let path = $file.as_path();

        File::open(path).map_err(|err| match err.kind() {
            std::io::ErrorKind::NotFound => RecorderError::FileNotFound(err.to_string()),
            _ => RecorderError::Unknown(err.to_string()),
        })
    }};
}

macro_rules! str2writer {
    (
        $file:expr,
        $ext:expr
    ) => {{
        $file.set_extension($ext);
        let path = $file.as_path();

        if path.exists() {
            log::info!("File exists, replacing");
            std::fs::remove_file(path).map_err(|err| RecorderError::Unknown(err.to_string()))?;
        }

        File::create(path).map_err(|err| match err.kind() {
            std::io::ErrorKind::NotFound => RecorderError::FileNotFound(err.to_string()),
            _ => RecorderError::Unknown(err.to_string()),
        })
    }};
}

impl Recorder for FileBinGzRecorder {
    type RecordArgs = PathBuf;
    type RecordOutput = ();
    type LoadArgs = PathBuf;

    fn record<Obj: Serialize + DeserializeOwned>(
        obj: Obj,
        mut file: PathBuf,
    ) -> Result<(), RecorderError> {
        let config = bin_config();
        let writer = str2writer!(file, Self::file_extension())?;
        let mut writer = GzEncoder::new(writer, Compression::default());

        bincode::serde::encode_into_std_write(&obj, &mut writer, config)
            .map_err(|err| RecorderError::Unknown(err.to_string()))?;

        Ok(())
    }

    fn load<Obj: Serialize + DeserializeOwned>(mut file: PathBuf) -> Result<Obj, RecorderError> {
        let reader = str2reader!(file, Self::file_extension())?;
        let mut reader = GzDecoder::new(reader);
        let state = bincode::serde::decode_from_std_read(&mut reader, bin_config())
            .map_err(|err| RecorderError::Unknown(err.to_string()))?;

        Ok(state)
    }
}

impl Recorder for FileBinRecorder {
    type RecordArgs = PathBuf;
    type RecordOutput = ();
    type LoadArgs = PathBuf;

    fn record<Obj: Serialize + DeserializeOwned>(
        obj: Obj,
        mut file: PathBuf,
    ) -> Result<(), RecorderError> {
        let config = bin_config();
        let mut writer = str2writer!(file, "bin")?;
        bincode::serde::encode_into_std_write(&obj, &mut writer, config)
            .map_err(|err| RecorderError::Unknown(err.to_string()))?;
        Ok(())
    }

    fn load<Obj: Serialize + DeserializeOwned>(mut file: PathBuf) -> Result<Obj, RecorderError> {
        let mut reader = str2reader!(file, "bin")?;
        let state = bincode::serde::decode_from_std_read(&mut reader, bin_config())
            .map_err(|err| RecorderError::Unknown(err.to_string()))?;
        Ok(state)
    }
}

impl Recorder for FileJsonGzRecorder {
    type RecordArgs = PathBuf;
    type RecordOutput = ();
    type LoadArgs = PathBuf;

    fn record<Obj: Serialize + DeserializeOwned>(
        obj: Obj,
        mut file: PathBuf,
    ) -> Result<(), RecorderError> {
        let writer = str2writer!(file, Self::file_extension())?;
        let writer = GzEncoder::new(writer, Compression::default());
        serde_json::to_writer(writer, &obj)
            .map_err(|err| RecorderError::Unknown(err.to_string()))?;

        Ok(())
    }

    fn load<Obj: Serialize + DeserializeOwned>(mut file: PathBuf) -> Result<Obj, RecorderError> {
        let reader = str2reader!(file, Self::file_extension())?;
        let reader = GzDecoder::new(reader);
        let state = serde_json::from_reader(reader)
            .map_err(|err| RecorderError::Unknown(err.to_string()))?;

        Ok(state)
    }
}

impl Recorder for FilePrettyJsonRecorder {
    type RecordArgs = PathBuf;
    type RecordOutput = ();
    type LoadArgs = PathBuf;

    fn record<Obj: Serialize + DeserializeOwned>(
        obj: Obj,
        mut file: PathBuf,
    ) -> Result<(), RecorderError> {
        let writer = str2writer!(file, Self::file_extension())?;
        serde_json::to_writer_pretty(writer, &obj)
            .map_err(|err| RecorderError::Unknown(err.to_string()))?;
        Ok(())
    }

    fn load<Obj: Serialize + DeserializeOwned>(mut file: PathBuf) -> Result<Obj, RecorderError> {
        let reader = str2reader!(file, Self::file_extension())?;
        let state = serde_json::from_reader(reader)
            .map_err(|err| RecorderError::Unknown(err.to_string()))?;

        Ok(state)
    }
}

impl Recorder for FileNamedMpkGzRecorder {
    type RecordArgs = PathBuf;
    type RecordOutput = ();
    type LoadArgs = PathBuf;

    fn record<Obj: Serialize + DeserializeOwned>(
        obj: Obj,
        mut file: PathBuf,
    ) -> Result<(), RecorderError> {
        let writer = str2writer!(file, Self::file_extension())?;
        let mut writer = GzEncoder::new(writer, Compression::default());
        rmp_serde::encode::write_named(&mut writer, &obj)
            .map_err(|err| RecorderError::Unknown(err.to_string()))?;

        Ok(())
    }

    fn load<Obj: Serialize + DeserializeOwned>(mut file: PathBuf) -> Result<Obj, RecorderError> {
        let reader = str2reader!(file, Self::file_extension())?;
        let reader = GzDecoder::new(reader);
        let state = rmp_serde::decode::from_read(reader)
            .map_err(|err| RecorderError::Unknown(err.to_string()))?;

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
        record::{InMemoryBinRecorder, Record, RecordSettings},
        TestBackend,
    };

    static FILE_PATH: &str = "/tmp/burn_test_file_recorder";

    #[test]
    fn test_can_save_and_load_jsongz_format() {
        test_can_save_and_load::<FileJsonGzRecorder>()
    }

    #[test]
    fn test_can_save_and_load_bin_format() {
        test_can_save_and_load::<FileBinRecorder>()
    }

    #[test]
    fn test_can_save_and_load_bingz_format() {
        test_can_save_and_load::<FileBinGzRecorder>()
    }

    #[test]
    fn test_can_save_and_load_pretty_json_format() {
        test_can_save_and_load::<FilePrettyJsonRecorder>()
    }

    #[test]
    fn test_can_save_and_load_mpkgz_format() {
        test_can_save_and_load::<FileNamedMpkGzRecorder>()
    }

    fn test_can_save_and_load<Recorder: FileRecorder>() {
        #[derive(Debug, Default)]
        struct TestRecordSettings<R> {
            phantom: PhantomData<R>,
        }

        impl<R: crate::record::Recorder> RecordSettings for TestRecordSettings<R> {
            type FloatElem = f32;
            type IntElem = i32;
            type Recorder = R;
        }

        let model_before = create_model();
        model_before
            .clone()
            .into_record()
            .record::<TestRecordSettings<Recorder>>(FILE_PATH.into())
            .unwrap();

        let model_after = create_model().load_record(
            nn::LinearRecord::load::<TestRecordSettings<Recorder>>(FILE_PATH.into()).unwrap(),
        );

        let model_bytes_before = model_before
            .into_record()
            .record::<TestRecordSettings<InMemoryBinRecorder>>(())
            .unwrap();
        let model_bytes_after = model_after
            .into_record()
            .record::<TestRecordSettings<InMemoryBinRecorder>>(())
            .unwrap();

        assert_eq!(model_bytes_after, model_bytes_before);
    }

    pub fn create_model() -> nn::Linear<TestBackend> {
        nn::LinearConfig::new(32, 32).with_bias(true).init()
    }
}
