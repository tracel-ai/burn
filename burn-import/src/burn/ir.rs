use std::collections::HashMap;

use proc_macro2::{Span, TokenStream};
use quote::quote;

use burn::{nn::conv::Conv2dConfig, tensor::DataSerialize};
use syn::Ident;

#[derive(Debug, Clone)]
pub enum Node {
    Matmul(Matmul),
    Conv2d(Conv2d),
}

#[derive(Debug, Clone)]
pub struct Matmul {
    pub lhs: TensorInput,
    pub rhs: TensorInput,
    pub output: TensorOutput,
}

#[derive(Debug, Clone)]
pub struct Conv2d {
    pub name_field: Ident,
    pub name_input: TensorInput,
    pub output: TensorOutput,
    pub data_weights: DataSerialize<f32>,
    pub data_bias: DataSerialize<f32>,
    pub config: Conv2dConfig,
}

#[derive(Debug, Clone)]
pub struct TensorInput {
    pub name: Ident,
    pub dim: usize,
    ref_count: usize,
}

#[derive(Debug, Clone)]
pub struct TensorOutput {
    pub name: Ident,
    pub dim: usize,
}

impl TensorInput {
    pub fn ref_count(&mut self, var_count: &mut HashMap<Ident, usize>) {
        if let Some(count) = var_count.get_mut(&self.name) {
            *count += 1;
            self.ref_count = *count;
        }
    }
}

impl TensorOutput {
    pub fn ref_count(&mut self, var_count: &mut HashMap<Ident, usize>) {
        if let Some(count) = var_count.get_mut(&self.name) {
            *count -= 1;
        }
    }
}

impl ToTokens for TensorInput {
    fn to_tokens(&self) -> TokenStream {
        let name = &self.name;

        if self.ref_count > 1 {
            quote! {
                #name.clone()
            }
        } else {
            quote! {
                #name
            }
        }
    }
}

#[derive(Default)]
pub struct NameGenerator {
    conv2d_count: usize,
    matmul_count: usize,
}

#[derive(Default)]
pub struct BurnImports {
    tensor: bool,
    conv2d: bool,
}

impl BurnImports {
    pub fn register_conv2d(&mut self) {
        self.conv2d = true;
    }

    pub fn register_tensor(&mut self) {
        self.tensor = true;
    }
}

impl NameGenerator {
    pub fn gen_conv2d(&mut self) -> Ident {
        self.conv2d_count += 1;
        let name = format!("conv2d_{}", self.conv2d_count);
        Ident::new(&name, Span::call_site())
    }

    pub fn gen_matmul(&mut self) -> Ident {
        self.matmul_count += 1;
        let name = format!("matmul_{}", self.matmul_count);
        Ident::new(&name, Span::call_site())
    }
}

pub trait ToTokens {
    fn to_tokens(&self) -> TokenStream;
}

pub trait TensorReferences {
    fn increate_input_ref_count(&mut self, names: &mut HashMap<Ident, usize>);
    fn decreate_output_ref_count(&mut self, names: &mut HashMap<Ident, usize>);
}

pub trait NodeCodegen: TensorReferences {
    fn output_type(&self) -> TokenStream;
    fn output_name(&self) -> Ident;
    fn input_def(&self) -> TokenStream;
    fn field_name(&self) -> Option<Ident>;
    fn new_body(&self) -> TokenStream;
    fn new_field(&self) -> TokenStream;
    fn forward(&self) -> TokenStream;
}

impl<const N: usize, T: Copy + quote::ToTokens> ToTokens for [T; N] {
    fn to_tokens(&self) -> TokenStream {
        let mut body = quote! {};

        for i in 0..N {
            let elem = self[i];
            body.extend(quote! {#elem,});
        }

        quote! {
            [#body]
        }
    }
}

impl Node {
    pub fn tensor_ref(&mut self, var_count: HashMap<String, usize>) {}
    pub fn output_type(&self) -> TokenStream {
        match self {
            Node::Matmul(ir) => {
                let dim_lhs = ir.lhs.dim;
                let dim_rhs = ir.rhs.dim;

                let dim = dim_rhs.max(dim_lhs);

                quote! {
                    Tensor<B, #dim>
                }
            }
            Node::Conv2d(_ir) => {
                quote! {
                    Tensor<B, 4>
                }
            }
        }
    }
    pub fn output_name(&self) -> Ident {
        match self {
            Node::Matmul(ir) => ir.output.name.clone(),
            Node::Conv2d(ir) => ir.output.name.clone(),
        }
    }
    pub fn input_definition(&self) -> TokenStream {
        match self {
            Node::Matmul(ir) => {
                let name_lhs = &ir.lhs.name;
                let name_rhs = &ir.rhs.name;
                let dim_lhs = ir.lhs.dim;
                let dim_rhs = ir.rhs.dim;

                quote! {
                    #name_lhs: Tensor<B, #dim_lhs>, #name_rhs: Tensor<B, #dim_rhs>
                }
            }
            Node::Conv2d(ir) => {
                let name = &ir.name_input.name;
                let dim = ir.output.dim;

                quote! {
                    #name: Tensor<B, #dim>
                }
            }
        }
    }
    pub fn field_name(&self) -> Option<Ident> {
        match self {
            Node::Matmul(_) => None,
            Node::Conv2d(ir) => Some(ir.name_field.clone()),
        }
    }
    pub fn init_with_body(&self) -> TokenStream {
        match self {
            Node::Matmul(_attrs) => {
                quote! {}
            }
            Node::Conv2d(attrs) => {
                let name = &attrs.name_field;
                let channels = attrs.config.channels.to_tokens();
                let kernel_size = attrs.config.kernel_size.to_tokens();
                let stride = attrs.config.stride.to_tokens();
                let dilation = attrs.config.dilation.to_tokens();
                let groups = attrs.config.groups;
                let bias = attrs.config.bias;

                quote! {
                    let #name = Conv2dConfig::new(#channels, #kernel_size)
                        .with_stride(#stride)
                        .with_dilation(#dilation)
                        .with_groups(#groups)
                        .with_bias(#bias)
                        .init_with(record.#name);
                }
            }
        }
    }
    pub fn gen_model_field(&self) -> TokenStream {
        match self {
            Node::Matmul(_attrs) => {
                quote! {}
            }
            Node::Conv2d(attrs) => {
                let name = &attrs.name_field;

                quote! {
                    #name: Conv2d<B>,
                }
            }
        }
    }

    pub fn gen_model_forward(&self) -> TokenStream {
        match self {
            Node::Matmul(attrs) => {
                let lhs = attrs.lhs.to_tokens();
                let rhs = attrs.rhs.to_tokens();
                let output = &attrs.output.name;

                quote! {
                    let #output = #lhs.matmul(#rhs);
                }
            }
            Node::Conv2d(attrs) => {
                let input = attrs.name_input.to_tokens();
                let output = &attrs.output.name;
                let field = &attrs.name_field;

                quote! {
                    let #output = self.#field.forward(#input);
                }
            }
        }
    }
}
