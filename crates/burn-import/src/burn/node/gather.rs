use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{BurnImports, ToTokens, Type};

use burn::record::PrecisionSettings;
use quote::quote;

#[derive(Debug, Clone)]
pub struct GatherNode {
    pub input: Type,
    pub index: Type,
    pub output: Type,
    pub dim: usize,
}

impl GatherNode {
    pub fn new(input: Type, index: Type, output: Type, dim: usize) -> Self {
        Self {
            input,
            index,
            output,
            dim,
        }
    }

    fn forward_shape_gather(
        &self,
        scope: &mut crate::burn::Scope,
        node_position: usize,
    ) -> proc_macro2::TokenStream {
        let input_shape = match &self.input {
            Type::Shape(in_shape) => in_shape,
            _ => unreachable!(),
        };

        match &self.output {
            Type::Scalar(scalar_type) => {
                // Gathering a single element from a shape produces a scalar
                let scalar_ty = scalar_type.ty();
                match &self.index {
                    Type::Scalar(idx_scalar) => {
                        let index = &idx_scalar.name;
                        let input_shape_name = &input_shape.name;
                        let output = &self.output.name();
                        // Handle negative indices properly for runtime scalars
                        quote! {
                            let actual_idx = if #index < 0 {
                                (#input_shape_name.len() as i64 + #index) as usize
                            } else {
                                #index as usize
                            };
                            let #output = #input_shape_name[actual_idx] as #scalar_ty;
                        }
                    }
                    _ => panic!(
                        "Gather from Shape to Scalar needs scalar index, got {:?}!",
                        self.index
                    ),
                }
            }
            Type::Shape(out_shape) => {
                match &self.index {
                    Type::Tensor(idx_tensor) => {
                        let index = scope.tensor_use_owned(idx_tensor, node_position);
                        let index_rank = idx_tensor.rank;
                        let output_rank = out_shape.rank;
                        let input_shape_name = &input_shape.name;
                        let output = &self.output.name();

                        if index_rank == 1 {
                            // Handle negative indices properly for runtime tensors
                            quote! {
                                let #output: [i64; #output_rank] = #index.to_data()
                                    .iter::<i64>()
                                    .map(|idx| {
                                        let actual_idx = if idx < 0 {
                                            (#input_shape_name.len() as i64 + idx) as usize
                                        } else {
                                            idx as usize
                                        };
                                        #input_shape_name[actual_idx]
                                    })
                                    .collect::<alloc::vec::Vec<_>>()
                                    .try_into()
                                    .unwrap();
                            }
                        } else {
                            panic!(
                                "Multi-dimensional indices for Shape gather should be 1-dimensional, but got rank {}",
                                index_rank
                            );
                        }
                    }
                    Type::Shape(idx_shape) => {
                        // Shape indices for gathering from Shape
                        let index_name = &idx_shape.name;
                        let output_rank = out_shape.rank;
                        let input_shape_name = &input_shape.name;
                        let output = &self.output.name();

                        // Handle negative indices properly for runtime shape indices
                        quote! {
                            let #output: [i64; #output_rank] = #index_name
                                .iter()
                                .map(|&idx| {
                                    let actual_idx = if idx < 0 {
                                        (#input_shape_name.len() as i64 + idx) as usize
                                    } else {
                                        idx as usize
                                    };
                                    #input_shape_name[actual_idx]
                                })
                                .collect::<alloc::vec::Vec<_>>()
                                .try_into()
                                .unwrap();
                        }
                    }
                    _ => panic!(
                        "Gather from Shape to Shape needs Tensor or Shape index, got {:?}!",
                        self.index
                    ),
                }
            }
            _ => panic!(
                "Gather from Shape input can only output Shape or Scalar, got {:?}!",
                self.output
            ),
        }
    }

    fn forward_tensor_gather(
        &self,
        scope: &mut crate::burn::Scope,
        node_position: usize,
    ) -> proc_macro2::TokenStream {
        let dim = self.dim.to_tokens();
        let input = match &self.input {
            Type::Tensor(in_tensor) => in_tensor,
            _ => unreachable!(),
        };
        let input_rank = input.rank;
        let input = scope.tensor_use_owned(input, node_position);
        let output = &self.output.name();

        match &self.output {
            Type::Scalar(scalar_type) => {
                // Gathering a single element from a tensor produces a scalar
                let scalar_ty = scalar_type.ty();
                match &self.index {
                    Type::Scalar(idx_scalar) => {
                        let index = &idx_scalar.name;
                        let output = &scalar_type.name;
                        quote! {
                            let indices = Tensor::<B, 1, _>::from_data([#index], &*self.device);
                            let selected = Tensor::select(#input, #dim, indices);
                            let #output = selected.into_scalar().elem::<#scalar_ty>();
                        }
                    }
                    _ => panic!(
                        "Gather from Tensor to Scalar needs scalar index, got {:?}!",
                        self.index
                    ),
                }
            }
            Type::Tensor(_) => {
                match &self.index {
                    Type::Scalar(idx_scalar) => {
                        // Use tensor.slice(...) with range syntax for more efficient gather operation
                        let index = &idx_scalar.name;
                        let output_rank = input_rank - 1;

                        // Generate slice ranges: s![.., index..index+1, ..] where the range is at position `dim`
                        let slice_args = (0..input_rank)
                            .map(|i| {
                                if i == self.dim {
                                    quote! { (#index as usize)..((#index as usize) + 1) }
                                } else {
                                    quote! { .. }
                                }
                            })
                            .collect::<Vec<_>>();

                        quote! {
                            let sliced = #input.slice(s![#(#slice_args),*]);
                            let #output = sliced.squeeze_dim::<#output_rank>(#dim);
                        }
                    }
                    Type::Tensor(idx_tensor) => {
                        let index = scope.tensor_use_owned(idx_tensor, node_position);
                        let index_rank = idx_tensor.rank;
                        let output_rank = index_rank + input_rank - 1;
                        let final_rank = output_rank.max(1); // Ensure minimum rank of 1

                        // Use proc_macro2::Literal to avoid usize suffix
                        let index_rank_lit = proc_macro2::Literal::usize_unsuffixed(index_rank);
                        let final_rank_lit = proc_macro2::Literal::usize_unsuffixed(final_rank);

                        quote! {
                            let #output = #input.take::<#index_rank_lit, #final_rank_lit>(#dim, #index);
                        }
                    }
                    Type::Shape(shape_type) => {
                        let shape_name = &shape_type.name;

                        // Shape array can be directly used to create tensor data
                        quote! {
                            let indices = Tensor::<B, 1, _>::from_data(#shape_name, &*self.device);
                            let #output = Tensor::select(#input, #dim, indices);
                        }
                    }
                    _ => panic!(
                        "Gather needs Scalar, Tensor, or Shape index, got {:?}!",
                        self.index
                    ),
                }
            }
            _ => panic!("Gather needs Tensor output, got {:?}!", self.output),
        }
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for GatherNode {
    fn output_types(&self) -> Vec<Type> {
        vec![self.output.clone()]
    }

    fn input_types(&self) -> Vec<crate::burn::Type> {
        // All indices are now runtime, so always return both input and index
        vec![self.input.clone(), self.index.clone()]
    }

    fn forward(
        &self,
        scope: &mut crate::burn::Scope,
        node_position: usize,
    ) -> proc_macro2::TokenStream {
        match &self.input {
            Type::Shape(_) => self.forward_shape_gather(scope, node_position),
            Type::Tensor(_) => self.forward_tensor_gather(scope, node_position),
            _ => panic!("Gather needs Tensor or Shape input, got {:?}!", self.input),
        }
    }

    fn into_node(self) -> super::Node<PS> {
        Node::Gather(self)
    }

    fn register_imports(&self, _imports: &mut BurnImports) {
        // s is already available in burn::prelude::*
        // Data is also available through burn::prelude::*
    }
}

impl OnnxIntoNode for GatherNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let onnx_ir::Node::Gather(n) = node else {
            panic!("Expected Gather node");
        };
        let input = Type::from(n.inputs.first().unwrap());
        let indices = Type::from(&n.inputs[1]);
        let output = Type::from(n.outputs.first().unwrap());

        // burn-import always deals with runtime indices
        // All indices (including from ONNX Constant nodes) are treated as runtime arguments
        Self::new(input, indices, output, n.config.axis)
    }
}
