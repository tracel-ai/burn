use onnx_ir::external_data::{ExternalDataInfo, is_external_data};
use onnx_ir::protos::{StringStringEntryProto, tensor_proto::DataLocation};
use protobuf::EnumOrUnknown;
use std::fs::File;
use std::io::Write;
use tempfile::TempDir;

#[test]
fn test_external_data_info_parsing() {
    let entries = vec![
        StringStringEntryProto {
            key: "location".to_string(),
            value: "weights.bin".to_string(),
            ..Default::default()
        },
        StringStringEntryProto {
            key: "offset".to_string(),
            value: "1024".to_string(),
            ..Default::default()
        },
        StringStringEntryProto {
            key: "length".to_string(),
            value: "4096".to_string(),
            ..Default::default()
        },
    ];

    let info = ExternalDataInfo::from_proto(&entries).unwrap();
    assert_eq!(info.location, "weights.bin");
    assert_eq!(info.offset, Some(1024));
    assert_eq!(info.length, Some(4096));
}

#[test]
fn test_is_external_data() {
    let external = EnumOrUnknown::new(DataLocation::EXTERNAL);
    assert!(is_external_data(external));

    let default = EnumOrUnknown::new(DataLocation::DEFAULT);
    assert!(!is_external_data(default));
}

#[test]
fn test_external_data_read() {
    let temp_dir = TempDir::new().unwrap();
    let weights_path = temp_dir.path().join("weights.bin");

    let weight_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let weight_bytes: Vec<u8> = weight_data.iter().flat_map(|f| f.to_le_bytes()).collect();

    let mut weight_file = File::create(&weights_path).unwrap();
    weight_file.write_all(&weight_bytes).unwrap();
    drop(weight_file);

    let entries = vec![
        StringStringEntryProto {
            key: "location".to_string(),
            value: "weights.bin".to_string(),
            ..Default::default()
        },
        StringStringEntryProto {
            key: "length".to_string(),
            value: weight_bytes.len().to_string(),
            ..Default::default()
        },
    ];

    let info = ExternalDataInfo::from_proto(&entries).unwrap();
    let loaded_data = info.read_data(temp_dir.path()).unwrap();

    assert_eq!(loaded_data.len(), weight_bytes.len());
    assert_eq!(loaded_data, weight_bytes);
}

#[test]
fn test_external_data_with_offset() {
    let temp_dir = TempDir::new().unwrap();
    let weights_path = temp_dir.path().join("weights_offset.bin");

    let padding = vec![0u8; 1024];
    let weight_data: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];
    let weight_bytes: Vec<u8> = weight_data.iter().flat_map(|f| f.to_le_bytes()).collect();

    let mut full_data = padding;
    full_data.extend_from_slice(&weight_bytes);

    let mut weight_file = File::create(&weights_path).unwrap();
    weight_file.write_all(&full_data).unwrap();
    drop(weight_file);

    let entries = vec![
        StringStringEntryProto {
            key: "location".to_string(),
            value: "weights_offset.bin".to_string(),
            ..Default::default()
        },
        StringStringEntryProto {
            key: "offset".to_string(),
            value: "1024".to_string(),
            ..Default::default()
        },
        StringStringEntryProto {
            key: "length".to_string(),
            value: weight_bytes.len().to_string(),
            ..Default::default()
        },
    ];

    let info = ExternalDataInfo::from_proto(&entries).unwrap();
    let loaded_data = info.read_data(temp_dir.path()).unwrap();

    assert_eq!(loaded_data.len(), weight_bytes.len());
    assert_eq!(loaded_data, weight_bytes);
}
