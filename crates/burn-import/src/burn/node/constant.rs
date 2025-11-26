use super::prelude::*;
use onnx_ir::ir::TensorDataExt;

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

                let (ty, init) = match &t.dtype {
                    dtype if dtype.is_int() || dtype.is_uint() => (
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
                    dtype if dtype.is_float() => (
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
                    dtype if dtype.is_bool() => (
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
                let data = match &t.dtype {
                    dtype if dtype.is_int() || dtype.is_uint() => {
                        tensor_data.clone().convert::<PS::IntElem>()
                    }
                    dtype if dtype.is_float() => tensor_data.clone().convert::<PS::FloatElem>(),
                    dtype if dtype.is_bool() => tensor_data.clone(),
                    _ => return S::serialize_none(serializer),
                };

                // Serialize using ParamSerde which handles any rank
                let param_serde = ParamSerde::new(ParamId::new().to_string(), data);
                param_serde.serialize(serializer)
            }
            _ => S::serialize_none(serializer),
        }
    }

    fn forward(&self, _scope: &mut super::super::scope::ScopeAtPosition<'_>) -> TokenStream {
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

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use insta::assert_snapshot;
    use onnx_ir::GraphState;
    use onnx_ir::ir::{
        ArgType, Argument, DType, TensorData, TensorDataExt, TensorType, ValueSource,
    };
    use onnx_ir::node::constant::ConstantNode;
    use std::cell::RefCell;
    use std::rc::Rc;

    /// Helper function to create a ConstantNode with tensor data for testing
    fn create_constant_node(
        name: &str,
        tensor_data: TensorData,
        output_ty: ArgType,
    ) -> ConstantNode {
        // Create GraphState and register the constant
        let mut graph_data = GraphState::new(&[], &[], &[], &[]);
        graph_data.register_test_constant("const_value".to_string(), tensor_data.clone());

        // Get the data_id from the registered constant
        let data_id = graph_data
            .get_constant_data_id("const_value")
            .expect("Test constant should have data_id");

        // Attach GraphState
        let graph_data_rc = Rc::new(RefCell::new(graph_data));

        // Determine input type based on tensor data
        let input_ty = if tensor_data.shape.is_empty() {
            ArgType::Scalar(tensor_data.elem_type())
        } else {
            ArgType::Tensor(TensorType {
                dtype: tensor_data.elem_type(),
                rank: tensor_data.shape.len(),
                static_shape: Some(tensor_data.shape.to_vec()),
            })
        };

        // Create input argument with Static value
        let mut input = Argument::new(String::new(), input_ty);
        input.value_source = ValueSource::Static(data_id);
        input.set_value_store(Some(graph_data_rc.clone()));

        // Create output argument
        let mut output = Argument::new(format!("{}_out", name), output_ty);
        output.value_source = ValueSource::Constant;
        output.set_value_store(Some(graph_data_rc));

        ConstantNode {
            name: name.to_string(),
            inputs: vec![input],
            outputs: vec![output],
        }
    }

    // ==================== Tensor Output Tests ====================

    #[test]
    fn test_constant_tensor_f32_rank2() {
        let data = TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
        let node = create_constant_node(
            "weights",
            data,
            ArgType::Tensor(TensorType {
                dtype: DType::F32,
                rank: 2,
                static_shape: Some(vec![2, 2]),
            }),
        );
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self) -> Tensor<B, 2> {
            let weights_out = self.weights.val();
            weights_out
        }
        ");
    }

    #[test]
    fn test_constant_tensor_f64_rank3() {
        let data = TensorData::new(vec![0.5f64; 8], vec![2, 2, 2]);
        let node = create_constant_node(
            "bias_tensor",
            data,
            ArgType::Tensor(TensorType {
                dtype: DType::F64,
                rank: 3,
                static_shape: Some(vec![2, 2, 2]),
            }),
        );
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self) -> Tensor<B, 3> {
            let bias_tensor_out = self.bias_tensor.val();
            bias_tensor_out
        }
        ");
    }

    #[test]
    fn test_constant_tensor_i32_rank1() {
        let data = TensorData::new(vec![10i32, 20, 30], vec![3]);
        let node = create_constant_node(
            "indices",
            data,
            ArgType::Tensor(TensorType {
                dtype: DType::I32,
                rank: 1,
                static_shape: Some(vec![3]),
            }),
        );
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self) -> Tensor<B, 1, Int> {
            let indices_out = self.indices.val();
            indices_out
        }
        ");
    }

    #[test]
    fn test_constant_tensor_i64_rank4() {
        let data = TensorData::new(vec![1i64; 16], vec![2, 2, 2, 2]);
        let node = create_constant_node(
            "shape_data",
            data,
            ArgType::Tensor(TensorType {
                dtype: DType::I64,
                rank: 4,
                static_shape: Some(vec![2, 2, 2, 2]),
            }),
        );
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self) -> Tensor<B, 4, Int> {
            let shape_data_out = self.shape_data.val();
            shape_data_out
        }
        ");
    }

    #[test]
    fn test_constant_tensor_bool_rank2() {
        let data = TensorData::new(vec![true, false, true, false], vec![2, 2]);
        let node = create_constant_node(
            "mask",
            data,
            ArgType::Tensor(TensorType {
                dtype: DType::Bool,
                rank: 2,
                static_shape: Some(vec![2, 2]),
            }),
        );
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self) -> Tensor<B, 2, Bool> {
            let mask_out = self.mask.val();
            mask_out
        }
        ");
    }

    // ==================== Scalar Output Tests ====================

    #[test]
    fn test_constant_scalar_f32() {
        let data = TensorData::new(vec![3.14f32], vec![]);
        let node = create_constant_node("pi", data, ArgType::Scalar(DType::F32));
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self) -> f32 {
            let pi_out = 3.14f32;
            pi_out
        }
        ");
    }

    #[test]
    fn test_constant_scalar_f64() {
        let data = TensorData::new(vec![2.718f64], vec![]);
        let node = create_constant_node("euler", data, ArgType::Scalar(DType::F64));
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self) -> f64 {
            let euler_out = 2.718f64;
            euler_out
        }
        ");
    }

    #[test]
    fn test_constant_scalar_i32() {
        let data = TensorData::new(vec![42i32], vec![]);
        let node = create_constant_node("answer", data, ArgType::Scalar(DType::I32));
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self) -> i32 {
            let answer_out = 42i32;
            answer_out
        }
        ");
    }

    #[test]
    fn test_constant_scalar_i64() {
        let data = TensorData::new(vec![1000i64], vec![]);
        let node = create_constant_node("count", data, ArgType::Scalar(DType::I64));
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self) -> i64 {
            let count_out = 1000i64;
            count_out
        }
        ");
    }

    #[test]
    fn test_constant_scalar_bool_true() {
        let data = TensorData::new(vec![true], vec![]);
        let node = create_constant_node("flag", data, ArgType::Scalar(DType::Bool));
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self) -> bool {
            let flag_out = true;
            flag_out
        }
        ");
    }

    #[test]
    fn test_constant_scalar_bool_false() {
        let data = TensorData::new(vec![false], vec![]);
        let node = create_constant_node("enabled", data, ArgType::Scalar(DType::Bool));
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self) -> bool {
            let enabled_out = false;
            enabled_out
        }
        ");
    }

    // ==================== Shape Output Tests ====================

    #[test]
    fn test_constant_shape_rank1() {
        let data = TensorData::new(vec![10i64], vec![1]);
        let node = create_constant_node("single_dim", data, ArgType::Shape(1));
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self) -> [i64; 1] {
            let single_dim_out: [i64; 1] = [10i64];
            single_dim_out
        }
        ");
    }

    #[test]
    fn test_constant_shape_rank2() {
        let data = TensorData::new(vec![5i64, 10], vec![2]);
        let node = create_constant_node("dims", data, ArgType::Shape(2));
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self) -> [i64; 2] {
            let dims_out: [i64; 2] = [5i64, 10i64];
            dims_out
        }
        ");
    }

    #[test]
    fn test_constant_shape_rank3() {
        let data = TensorData::new(vec![2i64, 3, 4], vec![3]);
        let node = create_constant_node("shape_vec", data, ArgType::Shape(3));
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self) -> [i64; 3] {
            let shape_vec_out: [i64; 3] = [2i64, 3i64, 4i64];
            shape_vec_out
        }
        ");
    }

    #[test]
    fn test_constant_shape_rank4() {
        let data = TensorData::new(vec![1i64, 2, 3, 4], vec![4]);
        let node = create_constant_node("full_shape", data, ArgType::Shape(4));
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self) -> [i64; 4] {
            let full_shape_out: [i64; 4] = [1i64, 2i64, 3i64, 4i64];
            full_shape_out
        }
        ");
    }
}
