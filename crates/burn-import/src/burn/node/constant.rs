use super::{Node, NodeCodegen};
use crate::burn::{ScalarKind, ScalarType, Scope, TensorType, ToTokens, Type};
use burn::{
    module::ParamId,
    record::{ParamSerde, PrecisionSettings},
    tensor::TensorData,
};
use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;
use serde::Serialize;

#[derive(Debug, Clone)]
pub struct ConstantNode {
    pub name: String,
    pub value: ConstantValue,
    pub output: Type,
}

#[derive(Debug, Clone, new)]
pub enum ConstantValue {
    /// Float constant.
    Float32(f32),
    Float64(f64),

    /// Integer constant.
    Int32(i32),
    Int64(i64),

    // Boolean constant.
    Bool(bool),

    /// Tensor constant.
    Tensor(TensorType, TensorData),
}

impl ConstantValue {
    pub fn ty_tokens(&self) -> TokenStream {
        match self {
            ConstantValue::Float32(_) => quote! { f32 },
            ConstantValue::Float64(_) => quote! { f64 },
            ConstantValue::Int32(_) => quote! { i32 },
            ConstantValue::Int64(_) => quote! { i64 },
            ConstantValue::Bool(_) => quote! { bool },
            ConstantValue::Tensor(tensor_type, _) => {
                let ty = tensor_type.ty();
                quote! { burn::module::Param<#ty>}
            }
        }
    }

    pub fn val_tokens(&self) -> TokenStream {
        match self {
            ConstantValue::Float32(val) => quote! { #val },
            ConstantValue::Float64(val) => quote! { #val },
            ConstantValue::Int32(val) => quote! { #val },
            ConstantValue::Int64(val) => quote! { #val },
            ConstantValue::Bool(val) => quote! { #val },
            ConstantValue::Tensor(_, _) => {
                panic!("Tensor constant is not assignable.")
            }
        }
    }
}

impl ConstantNode {
    pub fn new(name: String, value: ConstantValue, output: Type) -> Self {
        Self {
            name,
            value,
            output,
        }
    }
    pub fn constant_value_into_type(&self) -> Type {
        let name = Ident::new(self.name.as_str(), Span::call_site());
        match &self.value {
            ConstantValue::Float32(_) => Type::Scalar(ScalarType {
                name,
                kind: ScalarKind::Float32,
            }),
            ConstantValue::Float64(_) => Type::Scalar(ScalarType {
                name,
                kind: ScalarKind::Float64,
            }),
            ConstantValue::Int32(_) => Type::Scalar(ScalarType {
                name,
                kind: ScalarKind::Int32,
            }),
            ConstantValue::Int64(_) => Type::Scalar(ScalarType {
                name,
                kind: ScalarKind::Int64,
            }),
            ConstantValue::Bool(_) => Type::Scalar(ScalarType {
                name,
                kind: ScalarKind::Bool,
            }),

            ConstantValue::Tensor(tensor_type, _) => Type::Tensor(tensor_type.clone()),
        }
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for ConstantNode {
    fn output_types(&self) -> Vec<Type> {
        vec![self.output.clone()]
    }

    fn input_types(&self) -> Vec<Type> {
        vec![]
    }

    fn field_type(&self) -> Option<Type> {
        match &self.value {
            ConstantValue::Tensor(tensor_type, _) => Some(Type::Tensor(tensor_type.clone())),
            _ => None,
        }
    }

    fn field_init(&self) -> Option<TokenStream> {
        match &self.value {
            ConstantValue::Tensor(tensor_type, _) => {
                let ty = tensor_type.ty();
                let name = Ident::new(self.name.as_ref(), Span::call_site());
                let shape = tensor_type.clone().shape.unwrap().to_tokens();
                let dim = tensor_type.rank.to_tokens();

                match tensor_type.kind {
                    crate::burn::TensorKind::Int => Some(quote! {
                        let #name: burn::module::Param<#ty> = burn::module::Param::uninitialized(
                            burn::module::ParamId::new(),
                            move |device, _require_grad| Tensor::<B, #dim, Int>::zeros(#shape, &device),
                            device.clone(),
                            false
                        );
                    }),
                    crate::burn::TensorKind::Float => Some(quote! {
                        let #name: burn::module::Param<#ty> = burn::module::Param::uninitialized(
                            burn::module::ParamId::new(),
                            move |device, _require_grad| Tensor::<B, #dim>::zeros(#shape, &device),
                            device.clone(),
                            false,
                        );
                    }),
                    crate::burn::TensorKind::Bool => Some(quote! {
                        let #name: burn::module::Param<#ty> = burn::module::Param::uninitialized(
                            burn::module::ParamId::new(),
                            move |device, _require_grad| Tensor::<B, #dim, Bool>::empty(#shape, &device),
                            device.clone(),
                            false,
                        );
                    }),
                }
            }
            _ => None,
        }
    }

    fn forward(&self, _scope: &mut Scope, _node_position: usize) -> TokenStream {
        let name = Ident::new(self.name.as_ref(), Span::call_site());
        let output = self.output.name();

        match &self.value {
            ConstantValue::Tensor(_, _) => {
                quote! {
                    let #output = self.#name.val();
                }
            }
            _ => {
                let val = self.value.val_tokens();
                let ty = self.value.ty_tokens();

                quote! {
                    let #output: #ty = #val;
                }
            }
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Constant(self)
    }

    fn field_serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if let ConstantValue::Tensor(_, data) = &self.value {
            let data = data.clone().convert::<PS::FloatElem>();
            let data = ParamSerde::new(ParamId::new().to_string(), data);
            return data.serialize(serializer);
        }

        S::serialize_none(serializer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::burn::{
        ScalarKind, ScalarType, TensorType, graph::BurnGraph, node::test::assert_tokens,
    };
    use burn::record::FullPrecisionSettings;
    use burn::tensor::TensorData;
    use quote::ToTokens;

    fn expected_tokens_constant_scalar(
        ty: TokenStream,
        val: TokenStream,
        output: TokenStream,
    ) -> TokenStream {
        quote! {
            use burn::{
                module::Module,
                tensor::{backend::Backend, Tensor},
            };

            #[derive(Module, Debug)]
            pub struct Model<B: Backend> {
                phantom: core::marker::PhantomData<B>,
                device: burn::module::Ignored<B::Device>,
            }

            impl<B: Backend> Model<B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    Self {
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }

                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(&self) -> #ty {
                    let #output: #ty = #val;
                    #output
                }
            }
        }
    }

    fn assert_codegen_constant_scalar(constant: ConstantValue, scalar_kind: ScalarKind) {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();
        let val = constant.val_tokens();
        let ty = constant.ty_tokens();
        let output = Ident::new("output", Span::call_site());

        graph.register(ConstantNode::new(
            "constant_scalar".to_owned(),
            constant,
            Type::Scalar(ScalarType::new(output.to_string(), scalar_kind)),
        ));

        graph.register_input_output(vec![], vec![output.to_string()]);

        let expected = expected_tokens_constant_scalar(ty, val, output.to_token_stream());
        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_constant_scalar_float32() {
        assert_codegen_constant_scalar(ConstantValue::Float32(3.14f32), ScalarKind::Float32);
    }

    #[test]
    fn test_codegen_constant_scalar_float64() {
        assert_codegen_constant_scalar(
            ConstantValue::Float64(3.111_222_333_444_555_f64),
            ScalarKind::Float64,
        );
    }

    #[test]
    fn test_codegen_constant_scalar_int32() {
        assert_codegen_constant_scalar(ConstantValue::Int32(123i32), ScalarKind::Int32);
    }

    #[test]
    fn test_codegen_constant_scalar_int64() {
        assert_codegen_constant_scalar(ConstantValue::Int64(42i64), ScalarKind::Int64);
    }

    #[test]
    fn test_codegen_constant_scalar_bool() {
        assert_codegen_constant_scalar(ConstantValue::Bool(true), ScalarKind::Bool);
        assert_codegen_constant_scalar(ConstantValue::Bool(false), ScalarKind::Bool);
    }

    #[test]
    fn test_codegen_constant_tensor_float() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        let const_tensor = Ident::new("const_tensor", Span::call_site());
        let dimensions = 1;
        let shape = vec![4];
        let data = TensorData::from([2f32, 2f32, 2f32, 2f32]);
        let tensor_type = TensorType::new_float_with_shape(
            const_tensor.to_string(),
            dimensions,
            Some(shape.clone()),
        );
        let constant = ConstantValue::Tensor(tensor_type.clone(), data);

        graph.register(ConstantNode::new(
            const_tensor.to_string(),
            constant.clone(),
            Type::Tensor(TensorType::new_float_with_shape(
                "output",
                dimensions,
                Some(shape.clone()),
            )),
        ));

        graph.register_input_output(vec![], vec!["output".to_string()]);

        let expected = quote! {
            use burn::{
                module::Module,
                tensor::{backend::Backend, Tensor},
            };

            #[derive(Module, Debug)]
            pub struct Model<B: Backend> {
                const_tensor:  burn::module::Param<Tensor<B, 1>>,
                phantom: core::marker::PhantomData<B>,
                device: burn::module::Ignored<B::Device>,
            }

            impl<B: Backend> Model<B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    let const_tensor: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
                        burn::module::ParamId::new(),
                        move |device, _require_grad| Tensor::<B, 1>::zeros([4], &device),
                        device.clone(),
                        false
                    );

                    Self {
                        const_tensor,
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }

                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(&self) -> Tensor<B, 1> {
                    let output = self.const_tensor.val();
                    output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_constant_tensor_int() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        let const_tensor = Ident::new("const_tensor_int", Span::call_site());
        let dimensions = 1;
        let shape = vec![3];
        let data = TensorData::from([1i32, 2i32, 3i32]);
        let tensor_type = TensorType::new_int_with_shape(
            const_tensor.to_string(),
            dimensions,
            Some(shape.clone()),
        );
        let constant = ConstantValue::Tensor(tensor_type.clone(), data);

        graph.register(ConstantNode::new(
            const_tensor.to_string(),
            constant.clone(),
            Type::Tensor(TensorType::new_int_with_shape(
                "output",
                dimensions,
                Some(shape.clone()),
            )),
        ));

        graph.register_input_output(vec![], vec!["output".to_string()]);

        let expected = quote! {
            use burn::{
                module::Module,
                tensor::{backend::Backend, Tensor},
            };
            use burn::tensor::Int;

            #[derive(Module, Debug)]
            pub struct Model<B: Backend> {
                const_tensor_int: burn::module::Param<Tensor<B, 1, Int>>,
                phantom: core::marker::PhantomData<B>,
                device: burn::module::Ignored<B::Device>,
            }

            impl<B: Backend> Model<B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    let const_tensor_int: burn::module::Param<Tensor<B, 1, Int>> = burn::module::Param::uninitialized(
                        burn::module::ParamId::new(),
                        move |device, _require_grad| Tensor::<B, 1, Int>::zeros([3], &device),
                        device.clone(),
                        false
                    );

                    Self {
                        const_tensor_int,
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }

                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(&self) -> Tensor<B, 1, Int> {
                    let output = self.const_tensor_int.val();
                    output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_constant_tensor_bool() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        let const_tensor = Ident::new("const_tensor_3d", Span::call_site());
        let dimensions = 3;
        let shape = vec![1, 3, 2];
        let data = TensorData::from([[[true, false], [true, false], [true, false]]]);
        let tensor_type = TensorType::new_bool_with_shape(
            const_tensor.to_string(),
            dimensions,
            Some(shape.clone()),
        );
        let constant = ConstantValue::Tensor(tensor_type.clone(), data);

        graph.register(ConstantNode::new(
            const_tensor.to_string(),
            constant.clone(),
            Type::Tensor(TensorType::new_bool_with_shape(
                "output",
                dimensions,
                Some(shape.clone()),
            )),
        ));

        graph.register_input_output(vec![], vec!["output".to_string()]);

        let expected = quote! {
            use burn::{
                module::Module,
                tensor::{backend::Backend, Tensor},
            };
            use burn::tensor::Bool;

            #[derive(Module, Debug)]
            pub struct Model<B: Backend> {
                const_tensor_3d: burn::module::Param<Tensor<B, 3, Bool>>,
                phantom: core::marker::PhantomData<B>,
                device: burn::module::Ignored<B::Device>,
            }

            impl<B: Backend> Model<B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    let const_tensor_3d: burn::module::Param<Tensor<B, 3, Bool>> = burn::module::Param::uninitialized(
                        burn::module::ParamId::new(),
                        move |device, _require_grad| Tensor::<B, 3, Bool>::empty([1, 3, 2], &device),
                        device.clone(),
                        false
                    );

                    Self {
                        const_tensor_3d,
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }

                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(&self) -> Tensor<B, 3, Bool> {
                    let output = self.const_tensor_3d.val();
                    output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
