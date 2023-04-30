use super::{bin_config, BurnRecord, Record, RecordSettings, Recorder, RecorderError};
use flate2::{read::GzDecoder, write::GzEncoder, Compression};
use std::{fs::File, path::PathBuf};

/// Recorder trait specialized to save and load data to and from files.
pub trait FileRecorder<R: Record, S: RecordSettings>:
    Recorder<R, S, RecordArgs = PathBuf, RecordOutput = (), LoadArgs = PathBuf>
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

impl<R: Record, S: RecordSettings> FileRecorder<R, S> for FileBinGzRecorder {
    fn file_extension() -> &'static str {
        "bin.gz"
    }
}
impl<R: Record, S: RecordSettings> FileRecorder<R, S> for FileBinRecorder {
    fn file_extension() -> &'static str {
        "bin"
    }
}
impl<R: Record, S: RecordSettings> FileRecorder<R, S> for FileJsonGzRecorder {
    fn file_extension() -> &'static str {
        "json.gz"
    }
}
impl<R: Record, S: RecordSettings> FileRecorder<R, S> for FilePrettyJsonRecorder {
    fn file_extension() -> &'static str {
        "json"
    }
}

impl<R: Record, S: RecordSettings> FileRecorder<R, S> for FileNamedMpkGzRecorder {
    fn file_extension() -> &'static str {
        "mpk.gz"
    }
}

macro_rules! str2reader {
    (
        $file:expr
    ) => {{
        $file.set_extension(<Self as FileRecorder<R, S>>::file_extension());
        let path = $file.as_path();

        File::open(path).map_err(|err| match err.kind() {
            std::io::ErrorKind::NotFound => RecorderError::FileNotFound(err.to_string()),
            _ => RecorderError::Unknown(err.to_string()),
        })
    }};
}

macro_rules! str2writer {
    (
        $file:expr
    ) => {{
        $file.set_extension(<Self as FileRecorder<R, S>>::file_extension());
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

impl<R: Record, S: RecordSettings> Recorder<R, S> for FileBinGzRecorder {
    type RecordArgs = PathBuf;
    type RecordOutput = ();
    type LoadArgs = PathBuf;

    fn save_item(
        &self,
        item: BurnRecord<R::Item<S>>,
        mut file: Self::RecordArgs,
    ) -> Result<(), RecorderError> {
        let config = bin_config();
        let writer = str2writer!(file)?;
        let mut writer = GzEncoder::new(writer, Compression::default());

        bincode::serde::encode_into_std_write(&item, &mut writer, config)
            .map_err(|err| RecorderError::Unknown(err.to_string()))?;

        Ok(())
    }

    fn load_item(&self, mut file: Self::LoadArgs) -> Result<BurnRecord<R::Item<S>>, RecorderError> {
        let reader = str2reader!(file)?;
        let mut reader = GzDecoder::new(reader);
        let state = bincode::serde::decode_from_std_read(&mut reader, bin_config())
            .map_err(|err| RecorderError::Unknown(err.to_string()))?;

        Ok(state)
    }
}

impl<R: Record, S: RecordSettings> Recorder<R, S> for FileBinRecorder {
    type RecordArgs = PathBuf;
    type RecordOutput = ();
    type LoadArgs = PathBuf;

    fn save_item(
        &self,
        item: BurnRecord<R::Item<S>>,
        mut file: Self::RecordArgs,
    ) -> Result<(), RecorderError> {
        let config = bin_config();
        let mut writer = str2writer!(file)?;
        bincode::serde::encode_into_std_write(&item, &mut writer, config)
            .map_err(|err| RecorderError::Unknown(err.to_string()))?;
        Ok(())
    }

    fn load_item(&self, mut file: Self::LoadArgs) -> Result<BurnRecord<R::Item<S>>, RecorderError> {
        let mut reader = str2reader!(file)?;
        let state = bincode::serde::decode_from_std_read(&mut reader, bin_config())
            .map_err(|err| RecorderError::Unknown(err.to_string()))?;
        Ok(state)
    }
}

impl<R: Record, S: RecordSettings> Recorder<R, S> for FileJsonGzRecorder {
    type RecordArgs = PathBuf;
    type RecordOutput = ();
    type LoadArgs = PathBuf;

    fn save_item(
        &self,
        item: BurnRecord<R::Item<S>>,
        mut file: Self::RecordArgs,
    ) -> Result<(), RecorderError> {
        let writer = str2writer!(file)?;
        let writer = GzEncoder::new(writer, Compression::default());
        serde_json::to_writer(writer, &item)
            .map_err(|err| RecorderError::Unknown(err.to_string()))?;

        Ok(())
    }

    fn load_item(&self, mut file: Self::LoadArgs) -> Result<BurnRecord<R::Item<S>>, RecorderError> {
        let reader = str2reader!(file)?;
        let reader = GzDecoder::new(reader);
        let state = serde_json::from_reader(reader)
            .map_err(|err| RecorderError::Unknown(err.to_string()))?;

        Ok(state)
    }
}

impl<R: Record, S: RecordSettings> Recorder<R, S> for FilePrettyJsonRecorder {
    type RecordArgs = PathBuf;
    type RecordOutput = ();
    type LoadArgs = PathBuf;

    fn save_item(
        &self,
        item: BurnRecord<R::Item<S>>,
        mut file: Self::RecordArgs,
    ) -> Result<(), RecorderError> {
        let writer = str2writer!(file)?;
        serde_json::to_writer_pretty(writer, &item)
            .map_err(|err| RecorderError::Unknown(err.to_string()))?;
        Ok(())
    }

    fn load_item(&self, mut file: Self::LoadArgs) -> Result<BurnRecord<R::Item<S>>, RecorderError> {
        let reader = str2reader!(file)?;
        let state = serde_json::from_reader(reader)
            .map_err(|err| RecorderError::Unknown(err.to_string()))?;

        Ok(state)
    }
}

impl<R: Record, S: RecordSettings> Recorder<R, S> for FileNamedMpkGzRecorder {
    type RecordArgs = PathBuf;
    type RecordOutput = ();
    type LoadArgs = PathBuf;

    fn save_item(
        &self,
        item: BurnRecord<R::Item<S>>,
        mut file: Self::RecordArgs,
    ) -> Result<(), RecorderError> {
        let writer = str2writer!(file)?;
        let mut writer = GzEncoder::new(writer, Compression::default());
        rmp_serde::encode::write_named(&mut writer, &item)
            .map_err(|err| RecorderError::Unknown(err.to_string()))?;

        Ok(())
    }

    fn load_item(&self, mut file: Self::LoadArgs) -> Result<BurnRecord<R::Item<S>>, RecorderError> {
        let reader = str2reader!(file)?;
        let reader = GzDecoder::new(reader);
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
        record::{BytesBinRecorder, DefaultRecordSettings},
        TestBackend,
    };

    static FILE_PATH: &str = "/tmp/burn_test_file_recorder";

    #[test]
    fn test_can_save_and_load_jsongz_format() {
        test_can_save_and_load(FileJsonGzRecorder)
    }

    #[test]
    fn test_can_save_and_load_bin_format() {
        test_can_save_and_load(FileBinRecorder)
    }

    #[test]
    fn test_can_save_and_load_bingz_format() {
        test_can_save_and_load(FileBinGzRecorder)
    }

    #[test]
    fn test_can_save_and_load_pretty_json_format() {
        test_can_save_and_load(FilePrettyJsonRecorder)
    }

    #[test]
    fn test_can_save_and_load_mpkgz_format() {
        test_can_save_and_load(FileNamedMpkGzRecorder)
    }

    fn test_can_save_and_load<
        Recorder: FileRecorder<nn::LinearRecord<TestBackend>, DefaultRecordSettings>,
    >(
        recorder: Recorder,
    ) {
        let model_before = create_model();
        recorder
            .record(model_before.clone().into_record(), FILE_PATH.into())
            .unwrap();

        let model_after = create_model().load_record(recorder.load(FILE_PATH.into()).unwrap());

        let model_bytes_before =
            BytesBinRecorder::into_bytes::<_, DefaultRecordSettings>(model_before.into_record())
                .unwrap();
        let model_bytes_after =
            BytesBinRecorder::into_bytes::<_, DefaultRecordSettings>(model_after.into_record())
                .unwrap();

        assert_eq!(model_bytes_after, model_bytes_before);
    }

    pub fn create_model() -> nn::Linear<TestBackend> {
        nn::LinearConfig::new(32, 32).with_bias(true).init()
    }
}
