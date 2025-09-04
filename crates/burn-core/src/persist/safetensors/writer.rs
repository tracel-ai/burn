use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use hashbrown::HashMap;
use std::io::{Seek, Write};

use super::format::{SafetensorsError, SafetensorsHeader, TensorInfo, write_header_size};
use crate::persist::writer::{TensorWriter, WriterConfig, WriterError, WriterStats};
use crate::persist::{ModulePersist, TensorView};
use crate::tensor::backend::Backend;
use burn_tensor::TensorData;

/// Safetensors writer for efficient tensor serialization
pub struct SafetensorsWriter<W: Write + Seek> {
    writer: W,
    header: SafetensorsHeader,
    data_offset: usize,
    config: WriterConfig,
    stats: WriterStats,
}

impl<W: Write + Seek> SafetensorsWriter<W> {
    /// Create a new SafetensorsWriter
    pub fn new(writer: W) -> Self {
        Self {
            writer,
            header: SafetensorsHeader::new(),
            data_offset: 0,
            config: WriterConfig::default(),
            stats: WriterStats::default(),
        }
    }

    /// Add metadata to the safetensors file
    pub fn add_metadata(&mut self, key: String, value: String) -> &mut Self {
        self.header = self.header.clone().with_metadata(key, value);
        self
    }

    /// Write a module's tensors to the file
    pub fn write_module<B: Backend, M: ModulePersist<B>>(
        &mut self,
        module: &M,
    ) -> Result<WriterStats, SafetensorsError> {
        let views = module.collect();
        self.write_views(views)
    }

    /// Write filtered tensors from a module
    #[cfg(target_has_atomic = "ptr")]
    pub fn write_module_filtered<B: Backend, M: ModulePersist<B>>(
        &mut self,
        module: &M,
        patterns: &[&str],
    ) -> Result<WriterStats, SafetensorsError> {
        let views = module
            .collect_filtered(patterns)
            .map_err(|e| SafetensorsError::SerdeError(format!("Filter error: {:?}", e)))?;

        self.write_views(views)
    }

    /// Write tensor views directly
    pub fn write_views(
        &mut self,
        views: HashMap<String, TensorView>,
    ) -> Result<WriterStats, SafetensorsError> {
        // Collect metadata for all tensors
        let mut tensor_data: Vec<(String, TensorData)> = Vec::new();

        for (name, view) in views {
            let data = view.to_data();
            let start = self.data_offset;
            let end = start + data.bytes.len();

            let info = TensorInfo::new(data.dtype, data.shape.clone(), start, end);
            self.header.add_tensor(name.clone(), info);
            self.data_offset = end;

            tensor_data.push((name, data));
        }

        // Write header
        self.write_header()?;

        // Write tensor data
        for (_, data) in tensor_data {
            self.writer
                .write_all(&data.bytes)
                .map_err(|e| SafetensorsError::Io(format!("Failed to write tensor data: {}", e)))?;

            self.stats.tensors_written += 1;
            self.stats.bytes_written += data.bytes.len();
        }

        Ok(self.stats.clone())
    }

    /// Write the header to the file
    fn write_header(&mut self) -> Result<(), SafetensorsError> {
        // Serialize header to JSON
        let header_bytes = self.header.to_bytes()?;
        let header_size = header_bytes.len() as u64;

        // Write header size (8 bytes)
        self.writer
            .write_all(&write_header_size(header_size))
            .map_err(|e| SafetensorsError::Io(format!("Failed to write header size: {}", e)))?;

        // Write header JSON
        self.writer
            .write_all(&header_bytes)
            .map_err(|e| SafetensorsError::Io(format!("Failed to write header: {}", e)))?;

        self.stats.bytes_written += 8 + header_bytes.len();

        Ok(())
    }

    /// Finalize the writer and flush all data
    pub fn finish(mut self) -> Result<WriterStats, SafetensorsError> {
        self.writer
            .flush()
            .map_err(|e| SafetensorsError::Io(format!("Failed to flush: {}", e)))?;

        Ok(self.stats)
    }
}

/// Adapter to implement TensorWriter trait for SafetensorsWriter
pub struct SafetensorsTensorWriter<W: Write + Seek> {
    inner: Option<SafetensorsWriter<W>>,
    config: WriterConfig,
    pending_tensors: HashMap<String, TensorView>,
}

impl<W: Write + Seek> SafetensorsTensorWriter<W> {
    /// Create a new SafetensorsTensorWriter
    pub fn new(writer: W) -> Self {
        Self {
            inner: Some(SafetensorsWriter::new(writer)),
            config: WriterConfig::default(),
            pending_tensors: HashMap::new(),
        }
    }
}

impl<W: Write + Seek + Send> TensorWriter for SafetensorsTensorWriter<W> {
    fn write_tensor(&mut self, path: &str, view: &TensorView) -> Result<(), WriterError> {
        // Buffer tensors until finish is called
        self.pending_tensors.insert(path.to_string(), view.clone());
        Ok(())
    }

    fn write_batch(
        &mut self,
        tensors: HashMap<String, TensorView>,
    ) -> Result<WriterStats, WriterError> {
        // Add all tensors to pending
        self.pending_tensors.extend(tensors);
        Ok(WriterStats::default())
    }

    fn finish(&mut self) -> Result<(), WriterError> {
        if let Some(mut writer) = self.inner.take() {
            let tensors = core::mem::take(&mut self.pending_tensors);
            writer
                .write_views(tensors)
                .map_err(|e| WriterError::Other(format!("Safetensors error: {}", e)))?;

            writer
                .finish()
                .map_err(|e| WriterError::Other(format!("Failed to finish: {}", e)))?;
        }
        Ok(())
    }

    fn config(&self) -> &WriterConfig {
        &self.config
    }

    fn set_config(&mut self, config: WriterConfig) {
        self.config = config;
    }

    fn supports_compression(&self) -> bool {
        false // Safetensors doesn't support built-in compression
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate as burn;
    use crate::TestBackend;
    use crate::module::{Module, Param};
    use crate::persist::ModulePersist;
    use burn_tensor::Tensor;
    use std::io::Cursor;

    #[derive(Module, Debug)]
    struct TestModule<B: Backend> {
        weight: Param<Tensor<B, 2>>,
        bias: Param<Tensor<B, 1>>,
    }

    impl<B: Backend> TestModule<B> {
        fn new(device: &B::Device) -> Self {
            Self {
                weight: Param::from_data([[1.0, 2.0], [3.0, 4.0]], device),
                bias: Param::from_data([5.0, 6.0], device),
            }
        }
    }

    #[test]
    fn test_write_module() {
        let device = Default::default();
        let module = TestModule::<TestBackend>::new(&device);

        let mut buffer = Cursor::new(Vec::new());
        let mut writer = SafetensorsWriter::new(&mut buffer);

        let stats = writer.write_module(&module).unwrap();
        assert_eq!(stats.tensors_written, 2);

        let final_stats = writer.finish().unwrap();
        assert_eq!(final_stats.tensors_written, 2);
        assert!(final_stats.bytes_written > 0);
    }

    #[test]
    fn test_write_views() {
        let device = Default::default();
        let module = TestModule::<TestBackend>::new(&device);
        let views = module.collect();

        let mut buffer = Cursor::new(Vec::new());
        let mut writer = SafetensorsWriter::new(&mut buffer);

        let stats = writer.write_views(views).unwrap();
        assert_eq!(stats.tensors_written, 2);

        writer.finish().unwrap();

        // Verify the buffer contains data
        let data = buffer.into_inner();
        assert!(data.len() > 8); // At least header size + some data
    }

    #[test]
    fn test_tensor_writer_trait() {
        use crate::persist::writer::ModuleWriter;

        let device = Default::default();
        let module = TestModule::<TestBackend>::new(&device);

        let buffer = Cursor::new(Vec::new());
        let mut writer = SafetensorsTensorWriter::new(buffer);

        // SafetensorsTensorWriter buffers everything until finish, so stats will be default
        let stats = module.write_to(&mut writer).unwrap();
        // The actual write happens in finish(), not in write_to
        assert_eq!(stats.tensors_written, 0); // Default stats returned during buffering
    }

    #[test]
    fn test_with_metadata() {
        let device = Default::default();
        let module = TestModule::<TestBackend>::new(&device);

        let mut buffer = Cursor::new(Vec::new());
        let mut writer = SafetensorsWriter::new(&mut buffer);

        writer
            .add_metadata("framework".to_string(), "burn".to_string())
            .add_metadata("version".to_string(), "0.14.0".to_string());

        writer.write_module(&module).unwrap();
        writer.finish().unwrap();

        let data = buffer.into_inner();
        assert!(data.len() > 0);
    }
}
