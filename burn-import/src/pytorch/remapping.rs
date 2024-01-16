// use super::{converter::ModuleType, Converter};

// use lazy_static::lazy_static;
// use regex::Regex;

// lazy_static! {
//     static ref LINEAR_PATTERN: Regex = Regex::new("^(.+)\\.weight$").unwrap();
// }

// impl Converter {
//     fn add_tagged_module(&mut self, pattern: &str, module_type: ModuleType) -> &mut Self {
//         self.tagged_modules
//             .push((convert_to_regex(pattern), module_type));
//         self
//     }

//     /// Tag a module as a batch norm module
//     pub fn batch_norm_module(&mut self, name: &str) -> &mut Self {
//         self.add_tagged_module(name, ModuleType::BatchNorm)
//     }

//     /// Tag a module as a linear module
//     pub fn linear_module(&mut self, name: &str) -> &mut Self {
//         self.add_tagged_module(name, ModuleType::Linear)
//     }

//     /// Tag a module as a conv1d module
//     pub fn conv1d_module(&mut self, name: &str) -> &mut Self {
//         self.add_tagged_module(name, ModuleType::Conv1d)
//     }

//     /// Tag a module as a conv2d module
//     pub fn conv2d_module(&mut self, name: &str) -> &mut Self {
//         self.add_tagged_module(name, ModuleType::Conv2d)
//     }

//     /// Tag a module as a conv_transpose_1d module
//     pub fn conv_transpose_1d_module(&mut self, name: &str) -> &mut Self {
//         self.add_tagged_module(name, ModuleType::ConvTranspose1d)
//     }

//     /// Tag a module as a conv_transpose_2d module
//     pub fn conv_transpose_2d_module(&mut self, name: &str) -> &mut Self {
//         self.add_tagged_module(name, ModuleType::ConvTranspose2d)
//     }

//     /// Tag a module as an embedding module
//     pub fn embedding_module(&mut self, name: &str) -> &mut Self {
//         self.add_tagged_module(name, ModuleType::Embedding)
//     }

//     /// Tag a module as a group norm module
//     pub fn group_norm_module(&mut self, name: &str) -> &mut Self {
//         self.add_tagged_module(name, ModuleType::GroupNorm)
//     }

//     /// Tag a module as a layer norm module
//     pub fn layer_norm_module(&mut self, name: &str) -> &mut Self {
//         self.add_tagged_module(name, ModuleType::LayerNorm)
//     }

//     /// Remap the tensors according to the tagged modules
//     pub fn remap(
//         &mut self,
//         mut tensors: HashMap<String, CandleTensor>,
//     ) -> HashMap<String, CandleTensor> {
//         // Remap the keys
//         for (pattern, replacement) in self.key_remap.iter() {
//             let mut new_tensors = HashMap::new();
//             for (name, tensor) in tensors.iter() {
//                 let new_name = pattern.replace_all(name, replacement.as_str()).to_string();
//                 new_tensors.insert(new_name, tensor.clone());
//             }
//             tensors = new_tensors;
//         }

//         // Guess the module types if needed
//         if self.guess_module_type {
//             self.guess_module_types(&tensors);
//         }

//         // Remap the tensors according to the tagged modules
//         for (pattern, module_type) in self.tagged_modules.iter() {
//             module_type.remap(&mut tensors, pattern.clone());
//         }
//         tensors
//     }

//     /// Using heuristics, guess the module types of the tensors
//     pub fn guess_module_types(&mut self, tensors: &HashMap<String, CandleTensor>) {
//         // iterate over the tensors and try to guess the module type
//         for (name, tensor) in tensors.iter() {
//             if let Some(module_path) = is_linear(name, tensor, tensors) {
//                 // Add the module to the tagged modules
//                 self.add_tagged_module(format!("{}.*", module_path).as_str(), ModuleType::Linear);
//             }
//         }
//     }

//     /// Set key remapping.
//     ///
//     /// # Arguments
//     ///
//     /// * `regex` - The Regex pattern to be replaced.
//     /// * `replacement` - The pattern to replace with.
//     ///
//     /// See [Regex](https://docs.rs/regex/1.5.4/regex/#syntax) for the pattern syntax and
//     /// [Replacement](https://docs.rs/regex/latest/regex/struct.Regex.html#method.replace) for the
//     /// replacement syntax.
//     pub fn key_remap(&mut self, regex: &str, replacement: &str) -> &mut Self {
//         self.key_remap
//             .push((convert_to_regex(regex), replacement.into()));
//         self
//     }
// }

// /// Check if the tensor is a linear module
// ///
// /// A linear module is a 2D tensor with a bias of the same size
// fn is_linear(
//     name: &str,
//     tensor: &CandleTensor,
//     tensors: &HashMap<String, CandleTensor>,
// ) -> Option<String> {
//     // Check if the tensor is 2D
//     if tensor.dims().len() != 2 {
//         return None;
//     }

//     // Check if the module has a bias with the same size
//     LINEAR_PATTERN.captures(name).and_then(|caps| {
//         let (_full, [module_path]) = caps.extract();
//         let bias_name = format!("{}.bias", module_path);

//         tensors.get(&bias_name).and_then(|bias| {
//             if bias.dims().len() == 1 && bias.dims()[0] == tensor.dims()[0] {
//                 Some(module_path.to_string())
//             } else {
//                 None
//             }
//         })
//     })
// }

// impl ModuleType {
//     pub fn remap(&self, tensors: &mut HashMap<String, CandleTensor>, pattern: Regex) {
//         match self {
//             ModuleType::Linear => {
//                 for (name, tensor) in tensors.iter_mut() {
//                     if pattern.is_match(name) && name.ends_with(".weight") {
//                         // Make sure the tensor is 2D
//                         assert_eq!(tensor.dims().len(), 2);

//                         // Transpose the tensor
//                         *tensor = tensor.permute((1, 0)).unwrap();
//                     }
//                 }
//             }

//             _ => {}
//         }
//     }
// }

// fn convert_to_regex(pattern: &str) -> Regex {
//     Regex::new(&format!("^{}$", pattern)).unwrap()
// }
