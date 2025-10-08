use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

use crate::protos::{StringStringEntryProto, tensor_proto::DataLocation};
use protobuf::EnumOrUnknown;

/// Information about external data location and parameters
#[derive(Debug, Clone)]
pub struct ExternalDataInfo {
    /// Relative path to the external data file
    pub location: String,
    /// Byte offset in the external file (optional)
    pub offset: Option<u64>,
    /// Number of bytes to read (optional)
    pub length: Option<usize>,
    /// SHA1 checksum for verification (optional)
    pub checksum: Option<String>,
}

impl ExternalDataInfo {
    /// Parse external data information from ONNX StringStringEntryProto list
    pub fn from_proto(external_data: &[StringStringEntryProto]) -> Option<Self> {
        if external_data.is_empty() {
            return None;
        }

        let mut location = None;
        let mut offset = None;
        let mut length = None;
        let mut checksum = None;

        for entry in external_data {
            match entry.key.as_str() {
                "location" => location = Some(entry.value.clone()),
                "offset" => {
                    offset = entry.value.parse::<u64>().ok();
                }
                "length" => {
                    length = entry.value.parse::<usize>().ok();
                }
                "checksum" => checksum = Some(entry.value.clone()),
                _ => {
                    log::warn!("Unknown external_data key: {}", entry.key);
                }
            }
        }

        location.map(|loc| ExternalDataInfo {
            location: loc,
            offset,
            length,
            checksum,
        })
    }

    /// Read the external data from the file system
    ///
    /// # Arguments
    /// * `base_dir` - Base directory where the ONNX model file is located
    ///
    /// # Returns
    /// Raw bytes from the external data file
    pub fn read_data(&self, base_dir: &Path) -> Result<Vec<u8>, std::io::Error> {
        let file_path = base_dir.join(&self.location);

        log::info!(
            "Reading external data from: {} (offset: {:?}, length: {:?})",
            file_path.display(),
            self.offset,
            self.length
        );

        let mut file = File::open(&file_path).map_err(|e| {
            std::io::Error::new(
                e.kind(),
                format!(
                    "Failed to open external data file {}: {}",
                    file_path.display(),
                    e
                ),
            )
        })?;

        // Seek to offset if specified
        if let Some(offset) = self.offset {
            file.seek(SeekFrom::Start(offset))?;
        }

        // Read data
        let data = if let Some(length) = self.length {
            // Read specific number of bytes
            let mut buffer = vec![0u8; length];
            file.read_exact(&mut buffer)?;
            buffer
        } else {
            // Read all remaining bytes from current position
            let mut buffer = Vec::new();
            file.read_to_end(&mut buffer)?;
            buffer
        };

        // TODO: Verify checksum if provided
        if let Some(ref checksum) = self.checksum {
            log::debug!("Checksum verification not yet implemented: {}", checksum);
        }

        Ok(data)
    }
}

/// Check if a tensor uses external data storage
pub fn is_external_data(data_location: EnumOrUnknown<DataLocation>) -> bool {
    data_location.enum_value_or_default() == DataLocation::EXTERNAL
}
