use proc_macro::TokenStream;
use quote::{format_ident, quote};

use proc_macro2::TokenStream as TokenStream2;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{
    FnArg, GenericArgument, Ident, ItemStruct, ItemTrait, Meta, Pat, PathArguments, ReturnType,
    Token, TraitItem, Type, TypeParamBound, parse_macro_input,
};

/// # `backend_extension`
///
/// Attribute macro that generates dispatch glue for Burn backend extension traits.
///
/// ## Usage
///
/// ```rust,ignore
/// use burn_backend_extension::backend_extension;
///
/// #[backend_extension(Wgpu, Cuda, Cpu, Autodiff)]
/// pub trait MyExtension: Backend {
///     fn fused_matmul_add_relu(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>, bias: FloatTensor<Self>) -> FloatTensor<Self>;
///     fn custom_threshold(x: FloatTensor<Self>, threshold: f32) -> FloatTensor<Self>;
/// }
/// ```
///
/// ### What gets generated
///
/// - An `impl Trait for Dispatch` is generated automatically.
/// - Each method dispatches to the corresponding implementation for the listed backends.
/// - If `Autodiff` is specified, autodiff variants are also handled automatically.
/// - All other backends are left as `unimplemented!()`.
#[proc_macro_attribute]
pub fn backend_extension(attr: TokenStream, item: TokenStream) -> TokenStream {
    // Parse the backend list
    let backends = parse_macro_input!(attr as Backends);
    // Parse the trait definition
    let trait_def = parse_macro_input!(item as ItemTrait);

    // Lower to extension representation, then expand codegen
    let expanded = lower_extension(backends, &trait_def)
        .map(|ir| expand_extension(ir, trait_def))
        .unwrap_or_else(|err| err.to_compile_error());

    TokenStream::from(expanded)
}

#[derive(Debug, Clone)]
struct Backend {
    pub kind: BackendKind,
    pub cfg: Option<Meta>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BackendKind {
    Cpu,
    Cuda,
    Rocm,
    Metal,
    Vulkan,
    Wgpu,
    WebGpu,
    Flex,
    NdArray,
    LibTorch,
}

impl BackendKind {
    fn try_from(ident: &Ident) -> syn::Result<Self> {
        match ident.to_string().as_str() {
            "Cpu" => Ok(BackendKind::Cpu),
            "Cuda" => Ok(BackendKind::Cuda),
            "Wgpu" => Ok(BackendKind::Wgpu),
            "WebGpu" => Ok(BackendKind::WebGpu),
            "Metal" => Ok(BackendKind::Metal),
            "Rocm" => Ok(BackendKind::Rocm),
            "Vulkan" => Ok(BackendKind::Vulkan),
            "Flex" => Ok(BackendKind::Flex),
            "NdArray" => Ok(BackendKind::NdArray),
            "LibTorch" => Ok(BackendKind::LibTorch),
            other => Err(syn::Error::new_spanned(
                ident,
                format!("Unsupported backend `{}`", other),
            )),
        }
    }
}

struct Backends {
    concrete: Vec<Backend>,
    autodiff: (bool, Option<Meta>),
}

// Helper to parse backend idents w/ optional cfg
struct BackendArg {
    id: Ident,
    cfg: Option<Meta>,
}

impl Parse for BackendArg {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let id: Ident = input.parse()?;
        let cfg = if input.peek(Token![:]) {
            input.parse::<Token![:]>()?;

            // This parses cfg(feature = "...") or any other meta item
            let meta: syn::Meta = input.parse()?;
            Some(meta)
        } else {
            None
        };

        Ok(Self { id, cfg })
    }
}

impl Parse for Backends {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let args = Punctuated::<BackendArg, Token![,]>::parse_terminated(input)?;

        let mut concrete = vec![];
        let mut autodiff = (false, None);

        for arg in args {
            if arg.id == "Autodiff" {
                autodiff = (true, arg.cfg);
                continue;
            }

            concrete.push(Backend {
                kind: BackendKind::try_from(&arg.id)?,
                cfg: arg.cfg,
            });
        }

        Ok(Backends { concrete, autodiff })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TensorKind {
    Float,
    Int,
    Bool,
}

#[allow(clippy::large_enum_variant)]
enum ArgKind {
    Tensor(TensorKind),
    // Passthrough - unhandled by the macro
    Other(Type),
}

struct OperationArg {
    name: Ident,
    kind: ArgKind,
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
enum OutputKind {
    Tensor(TensorKind),
    Custom(Type),
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
enum OperationOutput {
    Tensor(TensorKind),
    Tuple(Vec<OutputKind>),
    Custom(Type),
}

struct Operation {
    name: Ident,
    inputs: Vec<OperationArg>,
    output: OperationOutput,
    asyncness: bool,
}

struct Extension {
    trait_name: Ident,
    backends: Backends,
    ops: Vec<Operation>,
}

impl TensorKind {
    fn from_type(ty: &Type) -> Option<Self> {
        match ty {
            // Handle `<Self as Backend>::<Primitive>`
            Type::Path(tp) if tp.qself.is_some() => {
                let last = tp.path.segments.last()?.ident.to_string();

                match last.as_str() {
                    "FloatTensorPrimitive" => Some(Self::Float),
                    "IntTensorPrimitive" => Some(Self::Int),
                    "BoolTensorPrimitive" => Some(Self::Bool),
                    _ => None,
                }
            }
            // Handle simple paths: Float, FloatTensor<Self>, burn::...::FloatTensor<Self>
            Type::Path(tp) => {
                let last = tp.path.segments.last()?.ident.to_string();

                match last.as_str() {
                    // Shorthand
                    "Float" => Some(Self::Float),
                    "Int" => Some(Self::Int),
                    "Bool" => Some(Self::Bool),

                    // Full tensor types
                    "FloatTensor" => Some(Self::Float),
                    "IntTensor" => Some(Self::Int),
                    "BoolTensor" => Some(Self::Bool),

                    // Associated primitive types
                    "FloatTensorPrimitive" => Some(Self::Float),
                    "IntTensorPrimitive" => Some(Self::Int),
                    "BoolTensorPrimitive" => Some(Self::Bool),

                    _ => None,
                }
            }

            // Handle references like &FloatTensor<Self>
            Type::Reference(r) => Self::from_type(&r.elem),

            // Handle parentheses `(FloatTensor<Self>)`
            Type::Paren(p) => Self::from_type(&p.elem),

            // TODO: option/containers already handled?
            _ => None,
        }
    }

    fn to_primitive_ty(self) -> TokenStream2 {
        match self {
            Self::Float => quote! { burn::backend::tensor::FloatTensor<Self> },
            Self::Int => quote! { burn::backend::tensor::IntTensor<Self> },
            Self::Bool => quote! { burn::backend::tensor::BoolTensor<Self> },
        }
    }

    fn variant(self) -> Ident {
        match self {
            Self::Float => format_ident!("Float"),
            Self::Int => format_ident!("Int"),
            Self::Bool => format_ident!("Bool"),
        }
    }

    fn unwrap_method(self) -> Ident {
        // e.g. tensor.float() (BackendTensor method)
        format_ident!("{}", format!("{:?}", self).to_lowercase())
    }
}

fn backend_to_ident(b: &Backend) -> Ident {
    // Convert the enum variant to a string first, then to an Ident
    format_ident!("{}", format!("{:?}", b.kind))
}

fn extract_future_output_type(ty: &Type) -> Option<&Type> {
    if let Type::ImplTrait(impl_trait) = ty {
        for bound in &impl_trait.bounds {
            if let TypeParamBound::Trait(trait_bound) = bound {
                let last_segment = trait_bound.path.segments.last()?;
                if last_segment.ident == "Future"
                    && let PathArguments::AngleBracketed(args) = &last_segment.arguments
                {
                    for arg in &args.args {
                        if let GenericArgument::AssocType(assoc) = arg
                            && assoc.ident == "Output"
                        {
                            return Some(&assoc.ty);
                        }
                    }
                }
            }
        }
    }
    None
}

fn lower_extension(attr: Backends, item: &ItemTrait) -> syn::Result<Extension> {
    let mut ops = Vec::new();

    for trait_item in &item.items {
        let TraitItem::Fn(f) = trait_item else {
            continue;
        };

        // Parse Inputs
        let mut inputs = Vec::new();
        for arg in &f.sig.inputs {
            let FnArg::Typed(pt) = arg else { continue };
            let name = match pt.pat.as_ref() {
                Pat::Ident(p) => p.ident.clone(),
                _ => return Err(syn::Error::new_spanned(&pt.pat, "Unsupported pattern")),
            };
            let kind = TensorKind::from_type(&pt.ty)
                .map(ArgKind::Tensor)
                .unwrap_or_else(|| ArgKind::Other((*pt.ty).clone()));
            inputs.push(OperationArg { name, kind });
        }

        // Parse outputs

        // Parse outputs
        let (actual_ty, is_async) = match &f.sig.output {
            ReturnType::Default => {
                return Err(syn::Error::new_spanned(
                    &f.sig.output,
                    "Operations must return a value",
                ));
            }
            ReturnType::Type(_, ty) => {
                // If it's `impl Future<Output = T>`, extract T and mark as async.
                // Otherwise, use the type as-is and check for `async fn`.
                if let Some(out_ty) = extract_future_output_type(ty) {
                    (out_ty, true)
                } else {
                    (ty.as_ref(), f.sig.asyncness.is_some())
                }
            }
        };

        let output = match actual_ty {
            // TODO: expand support for vec and maybe nested containers
            Type::Tuple(tup) => {
                let elements = tup
                    .elems
                    .iter()
                    .map(|elem| {
                        if let Some(kind) = TensorKind::from_type(elem) {
                            Ok(OutputKind::Tensor(kind))
                        } else {
                            Ok(OutputKind::Custom(elem.clone()))
                        }
                    })
                    .collect::<syn::Result<Vec<_>>>()?;
                OperationOutput::Tuple(elements)
            }
            ty if TensorKind::from_type(ty).is_some() => {
                OperationOutput::Tensor(TensorKind::from_type(ty).unwrap())
            }
            ty => {
                // ExtensionType
                OperationOutput::Custom(ty.clone())
            }
        };

        ops.push(Operation {
            name: f.sig.ident.clone(),
            inputs,
            output,
            asyncness: is_async,
        });
    }

    Ok(Extension {
        trait_name: item.ident.clone(),
        backends: attr,
        ops,
    })
}

fn expand_extension(ir: Extension, original_trait: ItemTrait) -> TokenStream2 {
    let trait_name = &ir.trait_name;

    // Generate Dispatch Implementation
    let dispatch_methods = ir.ops.iter().map(|op| gen_dispatch_method(&ir, op));

    quote! {
        #original_trait

        impl #trait_name for burn::backend::Dispatch {
            #( #dispatch_methods )*
        }
    }
}

fn gen_dispatch_method(ir: &Extension, op: &Operation) -> TokenStream2 {
    let name = &op.name;
    let has_ad = ir.backends.autodiff.0;
    let ad_cfg_attr = ir
        .backends
        .autodiff
        .1
        .as_ref()
        .map(|meta| quote! { #[#meta] });

    let maybe_async = if op.asyncness {
        quote! { async }
    } else {
        quote! {}
    };

    let sig_args = op.inputs.iter().map(|arg| {
        let name = &arg.name;
        match &arg.kind {
            ArgKind::Tensor(k) => {
                let ty = k.to_primitive_ty();
                quote! { #name: #ty }
            }
            ArgKind::Other(ty) => quote! { #name: #ty },
        }
    });

    let ret_ty = match &op.output {
        OperationOutput::Tensor(k) => k.to_primitive_ty(),
        OperationOutput::Tuple(elems) => {
            let types = elems.iter().map(|e| match e {
                OutputKind::Tensor(k) => k.to_primitive_ty(),
                OutputKind::Custom(ty) => quote! { #ty },
            });
            quote! { (#(#types),*) }
        }
        OperationOutput::Custom(ty) => quote! { #ty },
    };

    let tensor_inputs: Vec<_> = op
        .inputs
        .iter()
        .filter_map(|a| match &a.kind {
            ArgKind::Tensor(_) => Some(&a.name),
            _ => None,
        })
        .collect();

    let match_inputs = match tensor_inputs.len() {
        0 => quote! { () },
        1 => {
            let name = tensor_inputs[0];
            quote! { #name.kind }
        }
        _ => {
            let kinds = tensor_inputs.iter().map(|n| quote! { #n.kind });
            quote! { (#(#kinds),*) }
        }
    };

    let first_tensor = op.inputs.iter().find_map(|a| match &a.kind {
        ArgKind::Tensor(_) => Some(&a.name),
        _ => None,
    });

    let ckp_logic = if let Some(name) = first_tensor {
        quote! { let checkpointing = #name.checkpointing.clone(); }
    } else {
        quote! { let checkpointing = None; }
    };

    let concrete_arms = ir
        .backends
        .concrete
        .iter()
        .map(|b| gen_backend_arm(ir, op, b));
    let ad_arm = if has_ad {
        Some(gen_autodiff_arm(ir, op))
    } else {
        None
    };

    quote! {
        #maybe_async fn #name(#(#sig_args),*) -> #ret_ty {
            #ckp_logic
            match #match_inputs {
                #( #concrete_arms )*
                #ad_cfg_attr
                #ad_arm
                _ => unimplemented!("Backend not supported for custom op `{}`", stringify!(#name)),
            }
        }
    }
}

fn gen_backend_arm(ir: &Extension, op: &Operation, backend: &Backend) -> TokenStream2 {
    let b_ident = backend_to_ident(backend);
    let trait_name = &ir.trait_name;
    let fn_name = &op.name;

    // If any cfg(..) was specified to gate the backend
    let cfg_attr = backend.cfg.as_ref().map(|meta| quote! { #[#meta] });

    // Filter for tensor arguments only
    let tensor_args: Vec<_> = op
        .inputs
        .iter()
        .filter_map(|a| match &a.kind {
            ArgKind::Tensor(k) => Some((&a.name, k)),
            _ => None,
        })
        .collect();

    // Build the match pattern for the same backend
    // e.g., (DispatchTensorKind::Wgpu(lhs), DispatchTensorKind::Wgpu(rhs))
    let pattern = if tensor_args.len() == 1 {
        let (name, _) = tensor_args[0];
        quote! { burn::backend::DispatchTensorKind::#b_ident(#name) }
    } else {
        let pats = tensor_args.iter().map(|(name, _)| {
            quote! { burn::backend::DispatchTensorKind::#b_ident(#name) }
        });
        quote! { (#(#pats),*) }
    };

    // Unwrap inner kind: lhs.float(), rhs.int(), etc.
    let unwraps = tensor_args.iter().map(|(name, kind)| {
        let method = kind.unwrap_method();
        quote! { let #name = #name.#method(); }
    });

    // Call the method and wrap the result
    let call_args = op.inputs.iter().map(|a| &a.name);
    let maybe_await = if op.asyncness {
        quote! { .await }
    } else {
        quote! {}
    };

    let wrap_out = match &op.output {
        OperationOutput::Tensor(kind) => {
            let wrapped = gen_tensor_wrap(kind, quote! { _out }, &b_ident, false);
            quote! { burn::backend::DispatchTensor { kind: #wrapped, checkpointing } }
        }
        OperationOutput::Tuple(elems) => {
            let elements = elems.iter().enumerate().map(|(i, elem)| {
                let idx = syn::Index::from(i);
                match elem {
                    OutputKind::Tensor(kind) => {
                        let wrapped = gen_tensor_wrap(kind, quote! { _out.#idx }, &b_ident, false);
                        quote! { burn::backend::DispatchTensor { kind: #wrapped, checkpointing } }
                    }
                    OutputKind::Custom(_) => {
                        quote! {
                            burn::backend::ExtensionType::map_type(
                                _out.#idx,
                                |tensor| burn::backend::DispatchTensorKind::#b_ident(tensor),
                                checkpointing,
                            )
                        }
                    }
                }
            });
            quote! { (#(#elements),*) }
        }
        OperationOutput::Custom(_) => {
            quote! { burn::backend::ExtensionType::map_type(_out, |tensor| burn::backend::DispatchTensorKind::#b_ident(tensor), checkpointing) }
        }
    };

    quote! {
        #cfg_attr
        #pattern => {
            #(#unwraps)*
            let _out = <#b_ident as #trait_name>::#fn_name(#(#call_args),*)#maybe_await;
            #wrap_out
        }
    }
}

fn gen_autodiff_arm(ir: &Extension, op: &Operation) -> TokenStream2 {
    let trait_name = &ir.trait_name;
    let fn_name = &op.name;

    // Filter for tensor arguments only
    let tensor_args: Vec<_> = op
        .inputs
        .iter()
        .filter_map(|a| match &a.kind {
            ArgKind::Tensor(k) => Some((&a.name, k)),
            _ => None,
        })
        .collect();

    // Build the match pattern for the same backend wrapped by autodiff
    let inner_arms = ir.backends.concrete.iter().map(|backend| {
        let cfg_attr = backend.cfg.as_ref().map(|meta| quote! { #[#meta] });
        let b_ident = backend_to_ident(backend);

        let pattern = if tensor_args.len() == 1 {
            let (name, kind) = tensor_args[0];

            if *kind == TensorKind::Float {
                quote! {
                    burn::backend::DispatchTensorKind::Autodiff(#name)
                }
            } else {
                quote! {
                    burn::backend::DispatchTensorKind::#b_ident(#name)
                }
            }
        } else {
            let pats = tensor_args.iter().map(|(name, kind)| {
                if **kind == TensorKind::Float {
                    quote! {
                        burn::backend::DispatchTensorKind::Autodiff(#name)
                    }
                } else {
                    quote! {
                        burn::backend::DispatchTensorKind::#b_ident(#name)
                    }
                }
            });

            quote! { (#(#pats),*) }
        };

        let unwraps = tensor_args.iter().map(|(name, kind)| {
            if **kind == TensorKind::Float {
                quote! {
                    let #name = match *#name {
                        burn::backend::DispatchTensorKind::#b_ident(t) => t.autodiff(),
                        _ => unreachable!("Autodiff backend mismatch"),
                    };
                }
            } else {
                let method = kind.unwrap_method();
                quote! {
                    let #name = #name.#method();
                }
            }
        });

        let call_args = op.inputs.iter().map(|a| &a.name);
        let maybe_await = if op.asyncness {
            quote! { .await }
        } else {
            quote! {}
        };

        let wrap_out = match &op.output {
            OperationOutput::Tensor(kind) => {
                let wrapped = gen_tensor_wrap(kind, quote! { _out }, &b_ident, true);
                quote! { burn::backend::DispatchTensor { kind: #wrapped, checkpointing } }
            }
            OperationOutput::Tuple(elems) => {
                let elements = elems.iter().enumerate().map(|(i, elem)| {
                let idx = syn::Index::from(i);
                match elem {
                    OutputKind::Tensor(kind) => {
                        let wrapped = gen_tensor_wrap(kind, quote! { _out.#idx }, &b_ident, true);
                        quote! { burn::backend::DispatchTensor { kind: #wrapped, checkpointing } }
                    }
                    OutputKind::Custom(_) => {
                        quote! {
                            burn::backend::ExtensionType::map_type(
                                _out.#idx,
                                |tensor| match tensor {
                                    burn::backend::BackendTensor::Float(t) => {
                                        burn::backend::DispatchTensorKind::Autodiff(
                                            Box::new(burn::backend::DispatchTensorKind::#b_ident(
                                                burn::backend::BackendTensor::Autodiff(t)
                                            ))
                                        )
                                    }
                                    _ => burn::backend::DispatchTensorKind::#b_ident(tensor),
                                },
                                checkpointing,
                            )
                        }
                    }
                }
            });
                quote! { (#(#elements),*) }
            }
            OperationOutput::Custom(_) => {
                quote! { burn::backend::ExtensionType::map_type(_out, |tensor| match tensor {
                    burn::backend::BackendTensor::Float(t) => {
                        burn::backend::DispatchTensorKind::Autodiff(
                            Box::new(burn::backend::DispatchTensorKind::#b_ident(
                                burn::backend::BackendTensor::Autodiff(t)
                            ))
                        )
                    }
                }, checkpointing) }
            }
        };

        quote! {
            #cfg_attr
            #pattern => {
                #(#unwraps)*
                type _ADBackend = Autodiff<#b_ident>;
                let _out = <_ADBackend as #trait_name>::#fn_name(#(#call_args),*)#maybe_await;
                #wrap_out
            }
        }
    });

    quote! {
        #( #inner_arms )*
    }
}

fn gen_tensor_wrap(
    kind: &TensorKind,
    val: TokenStream2,
    b_ident: &Ident,
    is_ad: bool,
) -> TokenStream2 {
    let variant = kind.variant();
    if is_ad && *kind == TensorKind::Float {
        quote! {
            burn::backend::DispatchTensorKind::Autodiff(
                Box::new(burn::backend::DispatchTensorKind::#b_ident(
                    burn::backend::BackendTensor::Autodiff(#val)
                ))
            )
        }
    } else {
        quote! {
            burn::backend::DispatchTensorKind::#b_ident(
                burn::backend::BackendTensor::#variant(#val)
            )
        }
    }
}

/// Derive macro to implement `ExtensionType` for custom structures returned by backend extensions.
///
/// When a custom backend extension operation needs to return multiple tensors or a mix of tensors
/// and metadata (instead of a single tensor primitive or container of primitives), this macro automates
/// the process of wrapping the tensor primitives so they can cross the boundary into the `Dispatch` backend.
///
/// # Requirements
///
/// - The struct must have named fields.
/// - The struct must be generic over a `Backend` type parameter (`B`).
///
/// # Field Attributes
///
/// By default, the macro inspects the type of each field:
/// - **Tensor Primitives**: Automatically mapped.
/// - **Other types**: Passed through unmodified.
///
/// To nest another custom struct that also implements `ExtensionType`, you must annotate it with
/// `#[extension_type]` to tell the macro to traverse it recursively.
///
/// # Example
///
/// ```rust,ignore
/// #[derive(ExtensionType)]
/// pub struct OperationOutput<B: Backend> {
///     pub bool: BoolTensor<B>,
///     pub int: IntTensor<B>,
///     pub float: FloaTensor<B>,
///     pub count: usize, // Non-tensor field passes through automatically
/// }
/// ```
#[proc_macro_derive(ExtensionType, attributes(extension_type))]
pub fn derive_extension_output(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as ItemStruct);
    let name = &input.ident;

    // Extract generics for the trait implementation block
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    let fields_wrap = match &input.fields {
        syn::Fields::Named(fields) => {
            let field_mappings = fields.named.iter().map(|f| {
                let f_name = &f.ident;
                let is_ext = f.attrs.iter().any(|attr| attr.path().is_ident("extension_type"));

                if is_ext {
                    // It's a nested extension type
                    quote! {
                        #f_name: self.#f_name.map_type(
                            &map_kind,
                            checkpointing
                        ),
                    }
                }
                else if let Some(tensor_kind) = TensorKind::from_type(&f.ty) {
                    // Tensor primitive
                    let variant_ident = tensor_kind.variant();
                    quote! {
                        #f_name: burn::backend::DispatchTensor {
                            kind: map_kind(burn::backend::BackendTensor::#variant_ident(self.#f_name)),
                            checkpointing,
                        },
                    }
                } else {
                    // Passthrough
                    quote! { #f_name: self.#f_name, }
                }
             });

            quote! { #name { #( #field_mappings )* } }
        }
        _ => panic!("ExtensionType derive only supports structs with named fields"),
    };

    TokenStream::from(quote! {
        impl #impl_generics burn::backend::ExtensionType<B> for #name #ty_generics #where_clause {
            type Target = #name<burn::backend::Dispatch>;

            fn map_type<F>(
                self,
                map_kind: F,
                checkpointing: Option<burn::backend::CheckpointingStrategy>,
            ) -> Self::Target
            where
                F: Fn(burn::backend::BackendTensor<B>) -> burn::backend::DispatchTensorKind,
            {
                #fields_wrap
            }
        }
    })
}
