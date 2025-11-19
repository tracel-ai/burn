use super::{NodeCodegen, arg_to_ident};
use crate::burn::{Field, Scope, ToTokens};
use burn::record::PrecisionSettings;
use onnx_ir::Argument;
use onnx_ir::ir::{ArgType, TensorDataExt};
use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::node::constant::ConstantNode {
    fn inputs(&self) -> &[Argument] {
        // Constant has no runtime inputs - data comes from the input's value store
        &[]
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn field(&self) -> Option<Field> {
        // Only tensor constants need a field for storing the parameter
        let output = self.outputs.first().unwrap();
        match &output.ty {
            ArgType::Tensor(t) => {
                let name = Ident::new(&self.name, Span::call_site());
                let rank = t.rank.to_tokens();

                // Get tensor data from the input (which holds the constant value)
                let input = self.inputs.first().unwrap();
                let tensor_data = input.value().expect("Constant node must have tensor data");
                let shape = tensor_data.shape.to_tokens();

                let (ty, init) = match t.dtype {
                    onnx_ir::ir::DType::I32 | onnx_ir::ir::DType::I64 => (
                        quote! { burn::module::Param<Tensor<B, #rank, Int>> },
                        quote! {
                            let #name: burn::module::Param<Tensor<B, #rank, Int>> = burn::module::Param::uninitialized(
                                burn::module::ParamId::new(),
                                move |device, _require_grad| Tensor::<B, #rank, Int>::zeros(#shape, device),
                                device.clone(),
                                false,
                                #shape.into(),
                            );
                        },
                    ),
                    onnx_ir::ir::DType::F32 | onnx_ir::ir::DType::F64 => (
                        quote! { burn::module::Param<Tensor<B, #rank>> },
                        quote! {
                            let #name: burn::module::Param<Tensor<B, #rank>> = burn::module::Param::uninitialized(
                                burn::module::ParamId::new(),
                                move |device, _require_grad| Tensor::<B, #rank>::zeros(#shape, device),
                                device.clone(),
                                false,
                                #shape.into(),
                            );
                        },
                    ),
                    onnx_ir::ir::DType::Bool => (
                        quote! { burn::module::Param<Tensor<B, #rank, Bool>> },
                        quote! {
                            let #name: burn::module::Param<Tensor<B, #rank, Bool>> = burn::module::Param::uninitialized(
                                burn::module::ParamId::new(),
                                move |device, _require_grad| Tensor::<B, #rank, Bool>::empty(#shape, device),
                                device.clone(),
                                false,
                                #shape.into(),
                            );
                        },
                    ),
                    _ => (
                        quote! { burn::module::Param<Tensor<B, #rank>> },
                        quote! {
                            let #name: burn::module::Param<Tensor<B, #rank>> = burn::module::Param::uninitialized(
                                burn::module::ParamId::new(),
                                move |device, _require_grad| Tensor::<B, #rank>::zeros(#shape, device),
                                device.clone(),
                                false,
                                #shape.into(),
                            );
                        },
                    ),
                };
                Some(Field::new(self.name.clone(), ty, init))
            }
            _ => None,
        }
    }

    fn field_serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use burn::module::ParamId;
        use burn::record::ParamSerde;
        use serde::Serialize;

        let output = self.outputs.first().unwrap();

        match &output.ty {
            ArgType::Tensor(t) => {
                // Get tensor data from the input
                let input = self.inputs.first().unwrap();
                let tensor_data = input.value().expect("Constant node must have tensor data");

                // Convert to appropriate element type based on dtype
                let data = match t.dtype {
                    onnx_ir::ir::DType::I32 | onnx_ir::ir::DType::I64 => {
                        tensor_data.clone().convert::<PS::IntElem>()
                    }
                    onnx_ir::ir::DType::F32 | onnx_ir::ir::DType::F64 => {
                        tensor_data.clone().convert::<PS::FloatElem>()
                    }
                    onnx_ir::ir::DType::Bool => tensor_data.clone(),
                    _ => return S::serialize_none(serializer),
                };

                // Serialize using ParamSerde which handles any rank
                let param_serde = ParamSerde::new(ParamId::new().to_string(), data);
                param_serde.serialize(serializer)
            }
            _ => S::serialize_none(serializer),
        }
    }

    fn forward(&self, _scope: &mut Scope, _node_position: usize) -> TokenStream {
        let output = arg_to_ident(self.outputs.first().unwrap());
        let output_ty = &self.outputs.first().unwrap().ty;

        match output_ty {
            ArgType::Tensor(_) => {
                // For tensor constants, reference the field
                let name = Ident::new(&self.name, Span::call_site());
                quote! {
                    let #output = self.#name.val();
                }
            }
            ArgType::Scalar(elem_type) => {
                // For scalar constants, get the value from input and embed directly
                let input = self.inputs.first().unwrap();
                let tensor_data = input.value().expect("Constant node must have tensor data");

                let value = match elem_type {
                    onnx_ir::ir::DType::F32 => {
                        let val = tensor_data.as_slice::<f32>().unwrap()[0];
                        quote! { #val }
                    }
                    onnx_ir::ir::DType::F64 => {
                        let val = tensor_data.as_slice::<f64>().unwrap()[0];
                        quote! { #val }
                    }
                    onnx_ir::ir::DType::I32 => {
                        let val = tensor_data.as_slice::<i32>().unwrap()[0];
                        quote! { #val }
                    }
                    onnx_ir::ir::DType::I64 => {
                        let val = tensor_data.as_slice::<i64>().unwrap()[0];
                        quote! { #val }
                    }
                    onnx_ir::ir::DType::Bool => {
                        let val = tensor_data.as_slice::<bool>().unwrap()[0];
                        quote! { #val }
                    }
                    _ => panic!("Unsupported scalar type for constant"),
                };

                quote! {
                    let #output = #value;
                }
            }
            ArgType::Shape(rank) => {
                // For shape constants, get the shape values from input
                let input = self.inputs.first().unwrap();
                let tensor_data = input.value().expect("Constant node must have tensor data");
                let shape_vec = tensor_data.to_i64_vec().unwrap();

                let values: Vec<_> = shape_vec
                    .iter()
                    .map(|&v| {
                        let v_lit = proc_macro2::Literal::i64_suffixed(v);
                        quote! { #v_lit }
                    })
                    .collect();

                let rank_lit = proc_macro2::Literal::usize_unsuffixed(*rank);

                quote! {
                    let #output: [i64; #rank_lit] = [#(#values),*];
                }
            }
        }
    }
}
