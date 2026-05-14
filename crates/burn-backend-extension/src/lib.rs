use proc_macro::TokenStream;
use quote::{format_ident, quote};

use proc_macro2::TokenStream as TokenStream2;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{
    FnArg, Ident, ItemTrait, Meta, Pat, ReturnType, Token, TraitItem, Type, parse_macro_input,
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
    // metal/vulkan/wgpu could be merged eventually
    // Only required at this time due to different default elem types
    Metal,
    Vulkan,
    Wgpu,
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

struct Operation {
    name: Ident,
    inputs: Vec<OperationArg>,
    outputs: Vec<TensorKind>,
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

        // Parse Outputs (Support single type or tuple)
        let mut outputs = Vec::new();
        match &f.sig.output {
            ReturnType::Default => {}
            ReturnType::Type(_, ty) => {
                // TODO: expand support for vec, nested containers, and possibly custom structs
                // (requires though since we have no type information)
                if let syn::Type::Tuple(tup) = ty.as_ref() {
                    for elem in &tup.elems {
                        outputs.push(TensorKind::from_type(elem).ok_or_else(|| {
                            syn::Error::new_spanned(
                                elem,
                                "Tuple elements must be Float, Int, or Bool",
                            )
                        })?);
                    }
                } else {
                    outputs.push(TensorKind::from_type(ty).ok_or_else(|| {
                        syn::Error::new_spanned(
                            ty,
                            "Return must be Float, Int, Bool, or a tuple of them",
                        )
                    })?);
                }
            }
        }

        ops.push(Operation {
            name: f.sig.ident.clone(),
            inputs,
            outputs,
        });
    }

    Ok(Extension {
        trait_name: item.ident.clone(),
        backends: attr,
        ops,
    })
}

fn expand_extension(ir: Extension, mut original_trait: ItemTrait) -> TokenStream2 {
    let trait_name = &ir.trait_name;

    // Rewrite the original trait to use full Burn types
    for item in &mut original_trait.items {
        if let TraitItem::Fn(f) = item {
            for arg in &mut f.sig.inputs {
                if let FnArg::Typed(pt) = arg
                    && let Some(kind) = TensorKind::from_type(&pt.ty)
                {
                    let ty = kind.to_primitive_ty();
                    *pt.ty = syn::parse2(ty).unwrap();
                }
            }
            if let ReturnType::Type(_, ty) = &mut f.sig.output
                && let Some(kind) = TensorKind::from_type(ty)
            {
                let new_ty = kind.to_primitive_ty();
                **ty = syn::parse2(new_ty).unwrap();
            }
        }
    }

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

    // Signature generation
    let sig_args = op.inputs.iter().map(|arg| {
        let name = &arg.name;
        match &arg.kind {
            ArgKind::Tensor(k) => {
                let ty = k.to_primitive_ty();
                quote! { mut #name: #ty }
            }
            ArgKind::Other(ty) => quote! { #name: #ty },
        }
    });

    let ret_ty = match op.outputs.len() {
        0 => unimplemented!(), // TODO: error?
        1 => op.outputs[0].to_primitive_ty(),
        _ => {
            let types = op.outputs.iter().map(|k| k.to_primitive_ty());
            quote! { (#(#types),*) }
        }
    };

    // Match inputs
    let tensor_inputs: Vec<_> = op
        .inputs
        .iter()
        .filter_map(|a| match &a.kind {
            ArgKind::Tensor(_) => Some(&a.name),
            _ => None,
        })
        .collect();
    let match_inputs = match tensor_inputs.len() {
        0 => unimplemented!(), // TODO: error?
        1 => {
            let name = tensor_inputs[0];
            quote! { #name.kind }
        }
        _ => {
            let kinds = tensor_inputs.iter().map(|n| quote! { #n.kind });
            quote! { (#(#kinds),*) }
        }
    };

    // Checkpointing logic (when applicable, just check first tensor)
    let first_tensor = op.inputs.iter().find_map(|a| match &a.kind {
        ArgKind::Tensor(_) => Some(&a.name),
        _ => None,
    });
    let ckp_logic = if let Some(name) = first_tensor {
        quote! {
            let checkpointing = #name.checkpointing.clone();
        }
    } else {
        quote! {}
    };

    // Match arms for DispatchTensorKind::$Backends
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

    // Wrap the resulting tensor(s)
    // If the macro is in AD-mode, we include the field. If not, we don't.
    let wrap_output = |kinds_access: TokenStream2| {
        quote! {
            burn::backend::DispatchTensor {
                kind: #kinds_access,
                checkpointing: checkpointing.clone(), // Field always present, but None when not autodiff
            }
        }
    };

    let final_return = if op.outputs.len() == 1 {
        wrap_output(quote! { _kinds })
    } else {
        let wraps = op.outputs.iter().enumerate().map(|(i, _)| {
            let idx = syn::Index::from(i);
            wrap_output(quote! { _kinds.#idx })
        });
        quote! { (#(#wraps),*) }
    };

    quote! {
        fn #name(#(#sig_args),*) -> #ret_ty {
            #ckp_logic

            let _kinds = match #match_inputs {
                #( #concrete_arms )*
                #ad_cfg_attr
                #ad_arm
                _ => unimplemented!("Backend not supported for custom op `{}`", stringify!(#name)),
            };

            #final_return
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
    let wrap_out = gen_result_wrap(op, &b_ident, false);

    quote! {
        #cfg_attr
        #pattern => {
            #(#unwraps)*
            let _out = <#b_ident as #trait_name>::#fn_name(#(#call_args),*);
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
        let wrap_out = gen_result_wrap(op, &b_ident, true);

        quote! {
            #cfg_attr
            #pattern => {
                #(#unwraps)*
                type _ADBackend = Autodiff<#b_ident>;
                let _out = <_ADBackend as #trait_name>::#fn_name(#(#call_args),*);
                #wrap_out
            }
        }
    });

    quote! {
        #( #inner_arms )*
    }
}

fn gen_result_wrap(op: &Operation, b_ident: &Ident, is_ad: bool) -> TokenStream2 {
    let wrap_item = |kind: TensorKind, val: TokenStream2| {
        let variant = kind.variant();
        if is_ad && kind == TensorKind::Float {
            // Wrap as Autodiff(Backend(Autodiff(tensor)))
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
    };

    let wrapped_kinds = match op.outputs.len() {
        0 => quote! { () },
        1 => {
            let kind = wrap_item(op.outputs[0], quote! { _out });
            quote! { #kind }
        }
        _ => {
            let elements = op.outputs.iter().enumerate().map(|(i, &kind)| {
                let idx = syn::Index::from(i);
                wrap_item(kind, quote! { _out.#idx })
            });
            quote! { (#(#elements),*) }
        }
    };

    quote! { #wrapped_kinds }
}
