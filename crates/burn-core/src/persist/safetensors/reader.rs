use alloc::boxed::Box;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use hashbrown::HashMap;
use std::io::{Read, Seek, SeekFrom};

use super::format::{SafetensorsError, SafetensorsHeader, read_header_size};
use crate::persist::TensorView;
use burn_tensor::TensorData;

/// Safetensors reader for efficient tensor deserialization
pub struct SafetensorsReader<R: Read + Seek> {
    reader: R,
    header: SafetensorsHeader,
    header_size: usize,
}

impl<R: Read + Seek> SafetensorsReader<R> {
    /// Create a new SafetensorsReader
    pub fn new(mut reader: R) -> Result<Self, SafetensorsError> {
        // Read header size (8 bytes)
        let mut size_bytes = [0u8; 8];
        reader
            .read_exact(&mut size_bytes)
            .map_err(|e| SafetensorsError::Io(format!("Failed to read header size: {}", e)))?;

        let header_size = read_header_size(&size_bytes)? as usize;

        // Read header JSON
        let mut header_bytes = vec![0u8; header_size];
        reader
            .read_exact(&mut header_bytes)
            .map_err(|e| SafetensorsError::Io(format!("Failed to read header: {}", e)))?;

        let header = SafetensorsHeader::from_bytes(&header_bytes)?;

        // Validate all tensor offsets
        for (name, info) in &header.tensors {
            info.validate_offsets().map_err(|e| {
                SafetensorsError::InvalidFormat(format!(
                    "Invalid offsets for tensor {}: {}",
                    name, e
                ))
            })?;
        }

        Ok(Self {
            reader,
            header,
            header_size: 8 + header_size, // Size prefix + JSON header
        })
    }

    /// Get metadata from the header
    pub fn metadata(&self) -> Option<&HashMap<String, String>> {
        self.header.metadata.as_ref()
    }

    /// Read a specific tensor's data
    pub fn read_tensor(&mut self, name: &str) -> Result<TensorData, SafetensorsError> {
        let info = self
            .header
            .tensors
            .get(name)
            .ok_or_else(|| SafetensorsError::TensorNotFound(name.to_string()))?;

        // Calculate actual file offset (header size + data offset)
        let file_offset = self.header_size + info.data_offsets[0];
        let data_size = info.data_offsets[1] - info.data_offsets[0];

        // Seek to tensor data
        self.reader
            .seek(SeekFrom::Start(file_offset as u64))
            .map_err(|e| SafetensorsError::Io(format!("Failed to seek to tensor data: {}", e)))?;

        // Read tensor bytes
        let mut bytes = vec![0u8; data_size];
        self.reader
            .read_exact(&mut bytes)
            .map_err(|e| SafetensorsError::Io(format!("Failed to read tensor data: {}", e)))?;

        Ok(TensorData {
            bytes: burn_tensor::Bytes::from_bytes_vec(bytes),
            shape: info.shape.clone(),
            dtype: info.to_burn_dtype(),
        })
    }

    /// Create a lazy tensor view for a specific tensor
    fn create_tensor_view(&mut self, name: &str) -> Result<TensorView, SafetensorsError> {
        let info = self
            .header
            .tensors
            .get(name)
            .ok_or_else(|| SafetensorsError::TensorNotFound(name.to_string()))?
            .clone();

        // Calculate actual file offset
        let _file_offset = self.header_size + info.data_offsets[0];
        let data_size = info.data_offsets[1] - info.data_offsets[0];

        // Create a closure that will read the data when needed
        let view = TensorView::from_closure(Box::new(move || {
            // This closure captures the necessary information to read the tensor later
            // In a real implementation, we'd need to handle the reader access here
            // For now, we'll read immediately (can be optimized with Arc<Mutex<R>> or similar)

            // Create placeholder data - in production, this would actually read from the file
            TensorData {
                bytes: burn_tensor::Bytes::from_bytes_vec(vec![0u8; data_size]),
                shape: info.shape.clone(),
                dtype: info.to_burn_dtype(),
            }
        }));

        Ok(view)
    }
}

impl<R: Read + Seek> SafetensorsReader<R> {
    /// List all tensor names in the file
    pub fn list_tensors(&self) -> Vec<String> {
        self.header.tensors.keys().cloned().collect()
    }

    /// Read a single tensor as a TensorView
    pub fn read_tensor_view(&mut self, path: &str) -> Result<TensorView, SafetensorsError> {
        // For simplicity, we'll read the data immediately
        // In a production implementation, this should be truly lazy
        let data = self.read_tensor(path)?;
        Ok(TensorView::from_data(data))
    }

    /// Read all tensors as TensorViews
    pub fn read_all_views(&mut self) -> Result<HashMap<String, TensorView>, SafetensorsError> {
        let mut views = HashMap::new();
        let tensor_names: Vec<String> = self.header.tensors.keys().cloned().collect();

        for name in tensor_names {
            let view = self.read_tensor_view(&name)?;
            views.insert(name, view);
        }

        Ok(views)
    }
}

/// Memory-mapped reader for efficient large file handling
#[cfg(feature = "std")]
pub struct MemmapSafetensorsReader {
    mmap: memmap2::Mmap,
    header: SafetensorsHeader,
    header_size: usize,
}

#[cfg(feature = "std")]
impl MemmapSafetensorsReader {
    /// Create a memory-mapped reader from a file path
    pub fn from_file(path: &std::path::Path) -> Result<Self, SafetensorsError> {
        use std::fs::File;

        let file = File::open(path)
            .map_err(|e| SafetensorsError::Io(format!("Failed to open file: {}", e)))?;

        let mmap = unsafe {
            memmap2::MmapOptions::new()
                .map(&file)
                .map_err(|e| SafetensorsError::Io(format!("Failed to mmap file: {}", e)))?
        };

        Self::from_mmap(mmap)
    }

    /// Create from an existing memory map
    pub fn from_mmap(mmap: memmap2::Mmap) -> Result<Self, SafetensorsError> {
        // Read header size
        if mmap.len() < 8 {
            return Err(SafetensorsError::InvalidFormat(
                "File too small for header size".to_string(),
            ));
        }

        let header_size = read_header_size(&mmap[0..8])? as usize;

        if mmap.len() < 8 + header_size {
            return Err(SafetensorsError::InvalidFormat(
                "File too small for header".to_string(),
            ));
        }

        // Parse header
        let header = SafetensorsHeader::from_bytes(&mmap[8..8 + header_size])?;

        Ok(Self {
            mmap,
            header,
            header_size: 8 + header_size,
        })
    }

    /// Read tensor data directly from memory map
    pub fn read_tensor(&self, name: &str) -> Result<TensorData, SafetensorsError> {
        let info = self
            .header
            .tensors
            .get(name)
            .ok_or_else(|| SafetensorsError::TensorNotFound(name.to_string()))?;

        let file_offset = self.header_size + info.data_offsets[0];
        let data_size = info.data_offsets[1] - info.data_offsets[0];

        if file_offset + data_size > self.mmap.len() {
            return Err(SafetensorsError::InvalidFormat(format!(
                "Tensor {} data extends beyond file",
                name
            )));
        }

        // Copy data from mmap
        let bytes = self.mmap[file_offset..file_offset + data_size].to_vec();

        Ok(TensorData {
            bytes: burn_tensor::Bytes::from_bytes_vec(bytes),
            shape: info.shape.clone(),
            dtype: info.to_burn_dtype(),
        })
    }
}

#[cfg(feature = "std")]
impl MemmapSafetensorsReader {
    /// List all tensor names in the file
    pub fn list_tensors(&self) -> Vec<String> {
        self.header.tensors.keys().cloned().collect()
    }

    /// Read a single tensor as a TensorView
    pub fn read_tensor_view(&self, path: &str) -> Result<TensorView, SafetensorsError> {
        let data = self.read_tensor(path)?;
        Ok(TensorView::from_data(data))
    }

    /// Read all tensors as TensorViews
    pub fn read_all_views(&self) -> Result<HashMap<String, TensorView>, SafetensorsError> {
        let mut views = HashMap::new();
        let tensor_names: Vec<String> = self.header.tensors.keys().cloned().collect();

        for name in tensor_names {
            let view = self.read_tensor_view(&name)?;
            views.insert(name, view);
        }

        Ok(views)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate as burn;
    use crate::TestBackend;
    use crate::module::{Module, Param};
    use crate::persist::ModulePersist;
    use crate::persist::safetensors::SafetensorsWriter;
    use crate::tensor::backend::Backend;
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

        fn new_zeros(device: &B::Device) -> Self {
            Self {
                weight: Param::from_tensor(Tensor::zeros([2, 2], device)),
                bias: Param::from_tensor(Tensor::zeros([2], device)),
            }
        }
    }

    #[test]
    fn test_write_and_read() {
        let device = Default::default();
        let module = TestModule::<TestBackend>::new(&device);

        // Write to buffer
        let mut buffer = Cursor::new(Vec::new());
        {
            let mut writer = SafetensorsWriter::new(&mut buffer);
            writer.write_module(&module).unwrap();
            writer.finish().unwrap();
        }

        // Read back
        buffer.set_position(0);
        let mut reader = SafetensorsReader::new(buffer).unwrap();

        // List tensors
        let tensors = reader.list_tensors();
        assert_eq!(tensors.len(), 2);
        assert!(tensors.contains(&"weight".to_string()));
        assert!(tensors.contains(&"bias".to_string()));
    }

    #[test]
    fn test_round_trip() {
        let device = Default::default();
        let module1 = TestModule::<TestBackend>::new(&device);
        let mut module2 = TestModule::<TestBackend>::new_zeros(&device);

        // Export from module1
        let views = module1.collect();

        // Write to buffer
        let mut buffer = Cursor::new(Vec::new());
        {
            let mut writer = SafetensorsWriter::new(&mut buffer);
            writer.write_views(views).unwrap();
            writer.finish().unwrap();
        }

        // Read and import to module2
        buffer.set_position(0);
        let mut reader = SafetensorsReader::new(buffer).unwrap();
        let loaded_views = reader.read_all_views().unwrap();

        let result = module2.apply(loaded_views);
        assert!(result.is_success());
        assert_eq!(result.applied.len(), 2);
    }

    #[test]
    fn test_metadata() {
        let device = Default::default();
        let module = TestModule::<TestBackend>::new(&device);

        // Write with metadata
        let mut buffer = Cursor::new(Vec::new());
        {
            let mut writer = SafetensorsWriter::new(&mut buffer);
            writer.add_metadata("test_key".to_string(), "test_value".to_string());
            writer.write_module(&module).unwrap();
            writer.finish().unwrap();
        }

        // Read and check metadata
        buffer.set_position(0);
        let mut reader = SafetensorsReader::new(buffer).unwrap();

        let metadata = reader.metadata().unwrap();
        assert_eq!(metadata.get("test_key"), Some(&"test_value".to_string()));
    }
}
