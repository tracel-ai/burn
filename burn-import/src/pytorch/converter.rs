use std::collections::HashMap;

use burn::record::PrecisionSettings;

use candle_core::Tensor as CandleTensor;

use super::{reader::NestedValue, ser::Serializer, target_file::serialize_tensor};

// /// Type of the record to be saved.
// ///
// /// Only the record types that support name/value for struct fields can be used.
// /// Positional record types are not supported, such as Bincode, because the order of the fields
// /// dot not match the order of the fields of Pytorch's modules.
// ///
// #[derive(Debug, Clone, Default, Copy, ValueEnum)]
// // #[strum(ascii_case_insensitive)]
// pub enum RecordType {
//     /// Pretty JSON format (useful for debugging).
//     // #[strum(serialize = "json")]
//     PrettyJson,

//     /// Compressed Named MessagePack.
//     ///
//     /// Note: This may cause infinite build.
//     ///       See [#952 bug](https://github.com/tracel-ai/burn/issues/952).
//     NamedMpkGz,

//     /// Uncompressed Named MessagePack.
//     #[default]
//     NamedMpk,
// }

// /// Convert states from `.pt` or `.pth` files and save them to the `out_dir`.
// #[derive(Debug, Default)]
// pub struct Converter {
//     out_dir: Option<PathBuf>,
//     /// List of torch files to generate source code from.
//     inputs: Vec<PathBuf>,
//     development: bool,
//     pub(super) half_precision: bool,
//     pub(super) record_type: RecordType,
//     pub(super) key_remap: Vec<(Regex, String)>,
//     pub(super) tagged_modules: Vec<(Regex, ModuleType)>,
//     pub(super) guess_module_type: bool,
// }

// /// Type of the module used to tag incoming tensors for automatic conversion to burn module structures.
// ///
// /// These modules have weights and their names/shapes need to be converted to burn's structure.
// #[derive(Debug, Clone)]
// pub enum ModuleType {
//     BatchNorm,
//     Conv1d,
//     Conv2d,
//     ConvTranspose1d,
//     ConvTranspose2d,
//     Embedding,
//     GroupNorm,
//     LayerNorm,
//     Linear,
// }

// impl Converter {
//     /// Create a new `Converter`.
//     pub fn new() -> Self {
//         init_log().ok(); // Error when init multiple times are ignored.

//         Self {
//             guess_module_type: true,
//             ..Self::default()
//         }
//     }

//     /// Set output directory.
//     pub fn out_dir(&mut self, out_dir: &str) -> &mut Self {
//         self.out_dir = Some(Path::new(out_dir).into());
//         self
//     }

//     /// Add input file.
//     pub fn input(&mut self, input: &str) -> &mut Self {
//         self.inputs.push(Path::new(input).into());
//         self
//     }

//     /// Set development mode.
//     ///
//     /// If this is set to true, the generated model will be saved as `.graph.txt` files and model
//     /// states will be saved as `.json` file.
//     pub fn development(&mut self, development: bool) -> &mut Self {
//         self.development = development;
//         self
//     }

//     /// Run code generation.
//     ///
//     /// This function is intended to be called from `build.rs` script.
//     pub fn run_from_script(&mut self) {
//         self.run(true);
//     }

//     /// Run code generation.
//     ///
//     /// This function is intended to be called from CLI.
//     pub fn run_from_cli(&mut self) {
//         self.run(false);
//     }

//     /// Specify parameter precision to be saved.
//     ///
//     /// # Arguments
//     ///
//     /// * `half_precision` - If true, half precision is saved. Otherwise, full precision is saved.
//     pub fn half_precision(&mut self, half_precision: bool) -> &mut Self {
//         self.half_precision = half_precision;
//         self
//     }

//     /// Specify the type of the record to be saved.
//     ///
//     /// # Arguments
//     ///
//     /// * `record_type` - The type of the record to be saved.
//     pub fn record_type(&mut self, record_type: RecordType) -> &mut Self {
//         self.record_type = record_type;
//         self
//     }

//     /// Guess module type using heuristics.
//     pub fn guess_module_type(&mut self, guess_module_type: bool) -> &mut Self {
//         self.guess_module_type = guess_module_type;
//         self
//     }

//     /// Run code generation.
//     fn run(&mut self, is_build_script: bool) {
//         log::info!("Starting to convert Pytorch weights to Burn's format");

//         // prepend the out_dir to the cargo_out_dir if this is a build script
//         let out_dir = if is_build_script {
//             let cargo_out_dir = env::var("OUT_DIR").expect("OUT_DIR env is not set");
//             let mut path = PathBuf::from(cargo_out_dir);

//             // // Append the out_dir to the cargo_out_dir
//             path.push(self.out_dir.clone().unwrap());
//             path
//         } else {
//             self.out_dir.as_ref().expect("out_dir is not set").clone()
//         };

//         log::debug!("Output directory: {:?}", out_dir);

//         create_dir_all(&out_dir).unwrap();

//         for input in self.inputs.clone().iter() {
//             let file_name = input.file_stem().unwrap();
//             let out_file: PathBuf = out_dir.join(file_name);

//             log::info!("Converting {:?}", input);
//             log::debug!("Input file name: {:?}", file_name);
//             log::debug!("Output file: {:?}", out_file);

//             self.convert_model(input, out_file);
//         }

//         log::info!("Finished converting Pytorch weights to Burn's format");
//     }

//     /// Convert model weights from the `input` file and save them to the `out_file`.
//     fn convert_model(&mut self, input: &PathBuf, out_file: PathBuf) {
//         log::info!("Converting model weights from {:?}", input);
//         log::debug!("Development mode: {:?}", self.development);
//         log::debug!("Output file: {:?}", out_file);

//         // Read the pickle file and return a vector of Candle tensors
//         let tensors: HashMap<String, CandleTensor> =
//             pickle::read_all(input).unwrap().into_iter().collect();

//         // Remap the keys (replace the keys in the map with the new keys)
//         let remapped_tensor = self.remap(tensors);

//         // Convert the vector of Candle tensors to a nested map/vector of tensors
//         let map = reverse_flatten(remapped_tensor);

//         self.convert_to_record(map, out_file);

//         log::info!("Model weights are converted");
//     }
// }

/// Helper function to insert a value into a nested map/vector of tensors.
fn insert_nested_value(current: &mut NestedValue, keys: &[&str], value: NestedValue) {
    if keys.is_empty() {
        *current = value;
        return;
    }

    match current {
        NestedValue::Map(map) => {
            if !map.contains_key(keys[0]) {
                let next = if keys[1..]
                    .first()
                    .and_then(|k| k.parse::<usize>().ok())
                    .is_some()
                {
                    NestedValue::Vec(Vec::new())
                } else {
                    NestedValue::Map(HashMap::new())
                };
                map.insert(keys[0].to_string(), next);
            }
            insert_nested_value(map.get_mut(keys[0]).unwrap(), &keys[1..], value);
        }
        NestedValue::Vec(vec) => {
            let index = keys[0].parse::<usize>().unwrap();
            if index >= vec.len() {
                vec.resize_with(index + 1, || NestedValue::Map(HashMap::new()));
            }
            insert_nested_value(&mut vec[index], &keys[1..], value);
        }
        _ => panic!("Invalid structure encountered"),
    }
}

/// Convert a vector of Candle tensors to a nested map/vector of tensors.
pub fn reverse_flatten<PS: PrecisionSettings>(input: HashMap<String, CandleTensor>) -> NestedValue {
    let mut result = NestedValue::Map(HashMap::new());

    for (key, value) in input {
        let parts: Vec<&str> = key.split('.').collect();
        let st = serialize_tensor::<_, PS>(&value, Serializer::new()).unwrap();

        insert_nested_value(&mut result, &parts, st);
    }

    result
}
