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
///
/// Supported tensor argument/return types: `FloatTensor<Self>`, `IntTensor<Self>`,
/// `BoolTensor<Self>`, and `QuantizedTensor<Self>` (a QFloat tensor passed to the
/// backend still quantized — the op reads the packed values/scales directly instead
/// of going through a dequantize).
///
/// ### Struct-of-tensors inputs
///
/// A custom struct of tensor primitives (deriving [`ExtensionType`](macro@ExtensionType)) can be
/// passed as an **input** by marking the argument with `#[extension_type]`:
///
/// ```rust,ignore
/// #[derive(ExtensionType)]
/// pub struct Inputs<B: Backend> { pub lhs: FloatTensor<B>, pub rhs: FloatTensor<B> }
///
/// #[backend_extension(Wgpu)]
/// pub trait MyExtension: Backend {
///     fn fused(#[extension_type] inputs: Inputs<Self>, alpha: f32) -> FloatTensor<Self>;
/// }
/// ```
///
/// The macro selects the backend from a representative tensor field of the struct and unwraps the
/// dispatch form (`Inputs<Dispatch>`) back into the concrete `Inputs<Wgpu>` before calling the
/// backend impl. Struct inputs may be freely mixed with bare tensor inputs, and several struct
/// inputs are allowed, and the op may be combined with `Autodiff` (the op's own
/// `impl ... for Autodiff<B>` hand-writes the backward pass, as for bare-tensor autodiff ops).
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
    Remote,
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
            "Remote" => Ok(BackendKind::Remote),
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
    Quantized,
}

#[allow(clippy::large_enum_variant)]
enum ArgKind {
    Tensor(TensorKind),
    // A custom struct of tensor primitives, marked `#[extension_type]` on the argument. Unwrapped
    // back into the concrete backend's `Struct<B>` before the backend call via `ExtensionType`.
    ExtensionStruct(Type),
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
                    "QuantizedTensorPrimitive" => Some(Self::Quantized),
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
                    "Quantized" => Some(Self::Quantized),

                    // Full tensor types
                    "FloatTensor" => Some(Self::Float),
                    "IntTensor" => Some(Self::Int),
                    "BoolTensor" => Some(Self::Bool),
                    "QuantizedTensor" => Some(Self::Quantized),

                    // Associated primitive types
                    "FloatTensorPrimitive" => Some(Self::Float),
                    "IntTensorPrimitive" => Some(Self::Int),
                    "BoolTensorPrimitive" => Some(Self::Bool),
                    "QuantizedTensorPrimitive" => Some(Self::Quantized),

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
            Self::Quantized => quote! { burn::backend::tensor::QuantizedTensor<Self> },
        }
    }

    fn variant(self) -> Ident {
        match self {
            Self::Float => format_ident!("Float"),
            Self::Int => format_ident!("Int"),
            Self::Bool => format_ident!("Bool"),
            Self::Quantized => format_ident!("Quantized"),
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
            // An argument annotated with `#[extension_type]` is a custom struct of tensor
            // primitives (the input counterpart of the `#[derive(ExtensionType)]` output path).
            let is_ext = pt
                .attrs
                .iter()
                .any(|attr| attr.path().is_ident("extension_type"));
            let kind = if is_ext {
                validate_extension_struct_ty(&pt.ty)?;
                ArgKind::ExtensionStruct((*pt.ty).clone())
            } else if let Some(k) = TensorKind::from_type(&pt.ty) {
                ArgKind::Tensor(k)
            } else {
                ArgKind::Other((*pt.ty).clone())
            };
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

fn expand_extension(ir: Extension, mut original_trait: ItemTrait) -> TokenStream2 {
    let trait_name = &ir.trait_name;

    // `#[extension_type]` is a helper attribute understood only by this macro. Strip it from the
    // argument list before re-emitting the trait, otherwise rustc rejects it as an unknown attribute.
    for item in &mut original_trait.items {
        if let TraitItem::Fn(f) = item {
            for arg in &mut f.sig.inputs {
                if let FnArg::Typed(pt) = arg {
                    pt.attrs
                        .retain(|attr| !attr.path().is_ident("extension_type"));
                }
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

    let maybe_async = if op.asyncness {
        quote! { async }
    } else {
        quote! {}
    };

    let sig_args: Vec<_> = op
        .inputs
        .iter()
        .map(|arg| {
            let name = &arg.name;
            match &arg.kind {
                ArgKind::Tensor(k) => {
                    let ty = k.to_primitive_ty();
                    quote! { #name: #ty }
                }
                // Keep the original `Struct<Self>` type. Inside `impl Trait for Dispatch`, `Self`
                // resolves to `Dispatch`, so the incoming value is the dispatch form `Struct<Dispatch>`.
                ArgKind::ExtensionStruct(ty) => quote! { #name: #ty },
                ArgKind::Other(ty) => quote! { #name: #ty },
            }
        })
        .collect();

    let has_struct_input = op
        .inputs
        .iter()
        .any(|a| matches!(a.kind, ArgKind::ExtensionStruct(_)));

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

    let body = if has_struct_input {
        // At least one custom struct of tensors is passed as input (marked `#[extension_type]`).
        // This path also covers mixing structs with bare tensors and multiple struct inputs.
        gen_mixed_dispatch_body(ir, op)
    } else if tensor_inputs.is_empty() {
        // No tensor input to select the backend from (e.g. `fn load_data(i: usize) -> FloatTensor`).
        // There is nothing to match on, so this is only well-defined for a single backend — the
        // remote backend is the motivating case (`#[backend_extension(Remote)]`), where the op is
        // shipped to the server. Dispatch directly to that backend; reject the ambiguous cases.
        if has_ad {
            quote! { compile_error!("A backend extension operation with no tensor inputs can't be combined with `Autodiff` — there is no input tensor to carry the autodiff graph.") }
        } else if ir.backends.concrete.len() == 1 {
            let backend = &ir.backends.concrete[0];
            let call = gen_backend_call(ir, op, backend);
            match &backend.cfg {
                // Ungated backend: dispatch straight to it.
                None => quote! { #ckp_logic #call },
                // The single backend is `cfg`-gated. Mirror the match path: gate the call on the
                // backend's cfg and fall back to `unimplemented!` when it's compiled out, so the
                // method still has a valid body (instead of referencing a backend that doesn't
                // exist). `ckp_logic` lives inside the gated arm so its `checkpointing` binding
                // isn't left dangling (and untypeable) when the backend is compiled out.
                Some(meta) => quote! {
                    match () {
                        #[#meta]
                        () => { #ckp_logic #call }
                        #[allow(unreachable_patterns)]
                        _ => unimplemented!("Backend not supported for custom op `{}`", stringify!(#name)),
                    }
                },
            }
        } else {
            quote! { compile_error!("A backend extension operation with no tensor inputs must list exactly one backend (e.g. `#[backend_extension(Remote)]`), since there is no input tensor to select the backend from.") }
        }
    } else {
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
            #ckp_logic
            match #match_inputs {
                #( #concrete_arms )*
                #ad_cfg_attr
                #ad_arm
                _ => unimplemented!("Backend not supported for custom op `{}`", stringify!(#name)),
            }
        }
    };

    quote! {
        #maybe_async fn #name(#(#sig_args),*) -> #ret_ty {
            #body
        }
    }
}

fn gen_backend_arm(ir: &Extension, op: &Operation, backend: &Backend) -> TokenStream2 {
    let b_ident = backend_to_ident(backend);

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

    let call = gen_backend_call(ir, op, backend);

    quote! {
        #cfg_attr
        #pattern => { #call }
    }
}

/// Rewrite a struct type's single generic argument to `param`, e.g. `MyStruct<Self>` -> `MyStruct<Wgpu>`.
/// Used to name the concrete `Struct<B>` (and the dispatch `Struct<Dispatch>`) from the `Struct<Self>`
/// written in the trait signature.
fn struct_ty_with_param(ty: &Type, param: TokenStream2) -> TokenStream2 {
    if let Type::Path(tp) = ty {
        let mut path = tp.path.clone();
        if let Some(last) = path.segments.last_mut() {
            last.arguments = PathArguments::None;
        }
        quote! { #path<#param> }
    } else {
        // Not a path type; leave it untouched and let rustc surface a clear error.
        quote! { #ty }
    }
}

/// Validate the type of a `#[extension_type]`-marked argument, emitting a clear `compile_error!` for
/// the common misuses instead of letting them surface as obscure trait/type errors deeper in codegen.
///
/// The argument must be a struct with exactly one generic backend parameter (`MyStruct<Self>`), since
/// [`struct_ty_with_param`] rewrites that single parameter to the backend when unwrapping. Marking a
/// bare tensor, a non-path type, or a multi-parameter type is rejected here.
fn validate_extension_struct_ty(ty: &Type) -> syn::Result<()> {
    if TensorKind::from_type(ty).is_some() {
        return Err(syn::Error::new_spanned(
            ty,
            "`#[extension_type]` marks a struct of tensor primitives, not a tensor argument. Remove \
             the attribute to pass a plain tensor.",
        ));
    }

    let Type::Path(tp) = ty else {
        return Err(syn::Error::new_spanned(
            ty,
            "`#[extension_type]` requires a struct type with a single generic backend parameter, e.g. \
             `MyStruct<Self>`.",
        ));
    };

    let last = tp.path.segments.last().ok_or_else(|| {
        syn::Error::new_spanned(ty, "`#[extension_type]` type must be a named struct")
    })?;
    let type_args = match &last.arguments {
        PathArguments::AngleBracketed(args) => args
            .args
            .iter()
            .filter(|a| matches!(a, GenericArgument::Type(_)))
            .count(),
        _ => 0,
    };
    if type_args != 1 {
        return Err(syn::Error::new_spanned(
            ty,
            "`#[extension_type]` struct must have exactly one generic backend parameter, e.g. \
             `MyStruct<Self>`.",
        ));
    }

    Ok(())
}

/// Generate the dispatch body for an operation whose inputs include at least one `#[extension_type]`
/// struct. General "peek then unwrap" path: handles any mix of bare tensors and structs (including
/// several structs), for both concrete backends and (when `Autodiff` is listed) the autodiff wrapper.
///
/// The pure-tensor path selects the backend by destructuring every tensor in the match pattern,
/// which is impossible for a struct. Instead we:
/// 1. Peek a representative [`DispatchTensor`] (a bare tensor itself, or a struct's via
///    `ExtensionType::dispatch_repr`) and fold its `.kind` into a `(is_autodiff, backend_index)`
///    tag. Folding to a small value drops the borrow before any input is moved.
/// 2. In the matched arm, unwrap every input for the selected backend and re-wrap the output.
///
/// For an autodiff arm the target backend is `Autodiff<B>`: float tensors/fields unwrap through the
/// `Autodiff(Box(#b(BackendTensor::Autodiff(_))))` nesting into `FloatTensor<Autodiff<B>>`, while
/// int/bool/quantized ones stay plain (autodiff only tracks floats). The op's own
/// `impl ... for Autodiff<B>` still hand-writes the backward pass exactly as for a bare-tensor
/// autodiff op; the macro only routes and re-wraps.
fn gen_mixed_dispatch_body(ir: &Extension, op: &Operation) -> TokenStream2 {
    let name = &op.name;
    let has_ad = ir.backends.autodiff.0;
    // cfg gating the `Autodiff` entry itself (e.g. `Autodiff: cfg(feature = "autodiff")`). Every
    // generated autodiff arm must carry it, mirroring the pure-tensor path, so the arms vanish when
    // autodiff is compiled out — otherwise their `DispatchTensorKind::Autodiff` / `Autodiff<B>`
    // references (themselves feature-gated) would fail to compile.
    let ad_cfg_attr = ir
        .backends
        .autodiff
        .1
        .as_ref()
        .map(|meta| quote! { #[#meta] });

    // Representative `DispatchTensor` expression: a bare tensor input itself, or a struct's
    // representative field. Read twice below (for `.kind` selection and `.checkpointing`); both are
    // cheap accessors whose borrows end within their statement, before any input is moved.
    let repr_expr = op
        .inputs
        .iter()
        .find_map(|a| match &a.kind {
            ArgKind::Tensor(_) => {
                let n = &a.name;
                Some(quote! { #n })
            }
            ArgKind::ExtensionStruct(ty) => {
                let n = &a.name;
                let target_ty = struct_ty_with_param(ty, quote! { burn::backend::Dispatch });
                Some(quote! {
                    <#target_ty as burn::backend::ExtensionType<burn::backend::Dispatch>>::dispatch_repr(&#n)
                })
            }
            ArgKind::Other(_) => None,
        })
        // Only ever reached when a struct input exists, so a representative is always present.
        .expect("mixed dispatch requires at least one tensor-ish input");

    // Tag arms fold the representative kind into `(is_autodiff, backend_index)`. cfg-gated backends
    // drop the same arm from both the tag match and the dispatch match, keeping indices aligned.
    let concrete_tag_arms = ir.backends.concrete.iter().enumerate().map(|(i, backend)| {
        let cfg_attr = backend.cfg.as_ref().map(|meta| quote! { #[#meta] });
        let b_ident = backend_to_ident(backend);
        quote! {
            #cfg_attr
            burn::backend::DispatchTensorKind::#b_ident(_) => (false, #i),
        }
    });
    let ad_tag_arm = has_ad.then(|| {
        let inner = ir.backends.concrete.iter().enumerate().map(|(i, backend)| {
            let cfg_attr = backend.cfg.as_ref().map(|meta| quote! { #[#meta] });
            let b_ident = backend_to_ident(backend);
            quote! {
                #cfg_attr
                burn::backend::DispatchTensorKind::#b_ident(_) => #i,
            }
        });
        quote! {
            #ad_cfg_attr
            burn::backend::DispatchTensorKind::Autodiff(inner) => (true, match inner.as_ref() {
                #( #inner )*
                #[allow(unreachable_patterns)]
                _ => usize::MAX,
            }),
        }
    });

    let concrete_call_arms = ir.backends.concrete.iter().enumerate().map(|(i, backend)| {
        let cfg_attr = backend.cfg.as_ref().map(|meta| quote! { #[#meta] });
        let b_ident = backend_to_ident(backend);

        // Pull each bare tensor out of its `DispatchTensor` into a `BackendTensor<B>`, matching what
        // the pure-tensor path binds in its pattern. `gen_backend_call` then applies `.float()`/etc.
        let pre_extract = op.inputs.iter().filter_map(|a| match &a.kind {
            ArgKind::Tensor(_) => {
                let n = &a.name;
                Some(quote! {
                    let #n = match #n.kind {
                        burn::backend::DispatchTensorKind::#b_ident(bt) => bt,
                        #[allow(unreachable_patterns)]
                        _ => panic!(
                            "backend extension op received tensor inputs on mismatched backends; all inputs must be on the same backend"
                        ),
                    };
                })
            }
            _ => None,
        });

        let call = gen_backend_call(ir, op, backend);
        quote! {
            #cfg_attr
            (false, #i) => {
                #( #pre_extract )*
                #call
            }
        }
    });

    let ad_call_arms: Vec<_> = if has_ad {
        ir.backends
            .concrete
            .iter()
            .enumerate()
            .map(|(i, backend)| gen_mixed_ad_arm(ir, op, backend, i, &ad_cfg_attr))
            .collect()
    } else {
        Vec::new()
    };

    quote! {
        let checkpointing = #repr_expr.checkpointing.clone();
        let __burn_backend_tag: (bool, usize) = match &#repr_expr.kind {
            #ad_tag_arm
            #( #concrete_tag_arms )*
            #[allow(unreachable_patterns)]
            _ => (false, usize::MAX),
        };
        match __burn_backend_tag {
            #( #concrete_call_arms )*
            #( #ad_call_arms )*
            _ => unimplemented!("Backend not supported for custom op `{}`", stringify!(#name)),
        }
    }
}

/// Generate a single `(true, backend_index)` autodiff arm for the mixed dispatch path: unwrap every
/// input into its `Autodiff<B>` primitive, call `<Autodiff<B> as Trait>::op`, and re-wrap the output.
fn gen_mixed_ad_arm(
    ir: &Extension,
    op: &Operation,
    backend: &Backend,
    i: usize,
    ad_cfg_attr: &Option<TokenStream2>,
) -> TokenStream2 {
    let cfg_attr = backend.cfg.as_ref().map(|meta| quote! { #[#meta] });
    let b_ident = backend_to_ident(backend);
    let trait_name = &ir.trait_name;
    let fn_name = &op.name;
    let maybe_await = if op.asyncness {
        quote! { .await }
    } else {
        quote! {}
    };

    let unwraps = op.inputs.iter().filter_map(|a| match &a.kind {
        // Float tensor: peel the autodiff nesting to recover `FloatTensor<Autodiff<B>>`.
        ArgKind::Tensor(TensorKind::Float) => {
            let n = &a.name;
            Some(quote! {
                let #n = match #n.kind {
                    burn::backend::DispatchTensorKind::Autodiff(inner) => match *inner {
                        burn::backend::DispatchTensorKind::#b_ident(bt) => bt.autodiff(),
                        #[allow(unreachable_patterns)]
                        _ => panic!(
                            "backend extension op received tensor inputs on mismatched backends; all inputs must be on the same backend"
                        ),
                    },
                    #[allow(unreachable_patterns)]
                    _ => panic!(
                        "backend extension op mixes autodiff-tracked and untracked float tensors; all float inputs must share the same tracking"
                    ),
                };
            })
        }
        // Int/bool/quantized tensor: not autodiff-tracked, unwrap as usual.
        ArgKind::Tensor(kind) => {
            let n = &a.name;
            let method = kind.unwrap_method();
            Some(quote! {
                let #n = match #n.kind {
                    burn::backend::DispatchTensorKind::#b_ident(bt) => bt.#method(),
                    #[allow(unreachable_patterns)]
                    _ => unreachable!("tensor input routed to the wrong backend"),
                };
            })
        }
        // Struct: reconstruct `Struct<Autodiff<B>>`. The closure maps each field's dispatch kind to a
        // `BackendTensor<Autodiff<B>>` — float fields carry the autodiff nesting, others stay plain.
        ArgKind::ExtensionStruct(ty) => {
            let n = &a.name;
            let ad_ty = struct_ty_with_param(ty, quote! { Autodiff<#b_ident> });
            Some(quote! {
                let #n = <#ad_ty as burn::backend::ExtensionType<Autodiff<#b_ident>>>::map_from_dispatch(
                    #n,
                    |kind| match kind {
                        burn::backend::DispatchTensorKind::Autodiff(inner) => match *inner {
                            burn::backend::DispatchTensorKind::#b_ident(burn::backend::BackendTensor::Autodiff(t)) =>
                                burn::backend::BackendTensor::Float(t),
                            #[allow(unreachable_patterns)]
                            _ => panic!(
                                "backend extension op received tensor inputs on mismatched backends; all inputs must be on the same backend"
                            ),
                        },
                        burn::backend::DispatchTensorKind::#b_ident(bt) => match bt {
                            burn::backend::BackendTensor::Int(t) => burn::backend::BackendTensor::Int(t),
                            burn::backend::BackendTensor::Bool(t) => burn::backend::BackendTensor::Bool(t),
                            burn::backend::BackendTensor::Quantized(t) => burn::backend::BackendTensor::Quantized(t),
                            #[allow(unreachable_patterns)]
                            _ => unreachable!("autodiff struct non-float field unexpected variant"),
                        },
                        #[allow(unreachable_patterns)]
                        _ => panic!(
                            "backend extension op received tensor inputs on mismatched backends; all inputs must be on the same backend"
                        ),
                    },
                );
            })
        }
        ArgKind::Other(_) => None,
    });

    let call_args = op.inputs.iter().map(|a| &a.name);
    let wrap_out = gen_output_wrap(op, &b_ident, true);

    quote! {
        #ad_cfg_attr
        #cfg_attr
        (true, #i) => {
            #( #unwraps )*
            type _ADBackend = Autodiff<#b_ident>;
            let _out = <_ADBackend as #trait_name>::#fn_name(#( #call_args ),*)#maybe_await;
            #wrap_out
        }
    }
}

/// Wrap the backend call's result `_out` back into the dispatch representation. Shared by every
/// dispatch arm (concrete and autodiff). `is_ad` toggles the autodiff nesting: float outputs of an
/// `Autodiff<B>` op are re-wrapped as `Autodiff(Box(#b(BackendTensor::Autodiff(_))))`, while int/bool/
/// quantized outputs stay plain (autodiff only tracks floats).
fn gen_output_wrap(op: &Operation, b_ident: &Ident, is_ad: bool) -> TokenStream2 {
    // Wrap a single custom (`ExtensionType`) output value read via `accessor`.
    let wrap_custom = |accessor: TokenStream2| -> TokenStream2 {
        if is_ad {
            quote! {
                burn::backend::ExtensionType::map_to_dispatch(
                    #accessor,
                    |tensor| match tensor {
                        burn::backend::BackendTensor::Float(t) => burn::backend::DispatchTensorKind::Autodiff(
                            Box::new(burn::backend::DispatchTensorKind::#b_ident(
                                burn::backend::BackendTensor::Autodiff(t),
                            )),
                        ),
                        burn::backend::BackendTensor::Int(t) => burn::backend::DispatchTensorKind::#b_ident(burn::backend::BackendTensor::Int(t)),
                        burn::backend::BackendTensor::Bool(t) => burn::backend::DispatchTensorKind::#b_ident(burn::backend::BackendTensor::Bool(t)),
                        burn::backend::BackendTensor::Quantized(t) => burn::backend::DispatchTensorKind::#b_ident(burn::backend::BackendTensor::Quantized(t)),
                        #[allow(unreachable_patterns)]
                        _ => unreachable!("unexpected output tensor variant"),
                    },
                    checkpointing,
                )
            }
        } else {
            quote! {
                burn::backend::ExtensionType::map_to_dispatch(
                    #accessor,
                    |tensor| burn::backend::DispatchTensorKind::#b_ident(tensor),
                    checkpointing,
                )
            }
        }
    };

    match &op.output {
        OperationOutput::Tensor(kind) => {
            let wrapped = gen_tensor_wrap(kind, quote! { _out }, b_ident, is_ad);
            quote! { burn::backend::DispatchTensor { kind: #wrapped, checkpointing } }
        }
        OperationOutput::Tuple(elems) => {
            let elements = elems.iter().enumerate().map(|(i, elem)| {
                let idx = syn::Index::from(i);
                match elem {
                    OutputKind::Tensor(kind) => {
                        let wrapped = gen_tensor_wrap(kind, quote! { _out.#idx }, b_ident, is_ad);
                        quote! { burn::backend::DispatchTensor { kind: #wrapped, checkpointing } }
                    }
                    OutputKind::Custom(_) => wrap_custom(quote! { _out.#idx }),
                }
            });
            quote! { (#(#elements),*) }
        }
        OperationOutput::Custom(_) => wrap_custom(quote! { _out }),
    }
}

/// Generate the body that unwraps the dispatch tensors, calls the backend's trait impl and wraps the
/// result back into a [`DispatchTensor`]. Shared by [`gen_backend_arm`] (inside a match) and the
/// no-tensor-input dispatch path (direct call).
fn gen_backend_call(ir: &Extension, op: &Operation, backend: &Backend) -> TokenStream2 {
    let b_ident = backend_to_ident(backend);
    let trait_name = &ir.trait_name;
    let fn_name = &op.name;

    // Unwrap inner kind: lhs.float(), rhs.int(), etc. (no-op when there are no tensor inputs).
    let unwraps = op.inputs.iter().filter_map(|a| match &a.kind {
        ArgKind::Tensor(kind) => {
            let name = &a.name;
            let method = kind.unwrap_method();
            Some(quote! { let #name = #name.#method(); })
        }
        // Reconstruct `Struct<B>` from the incoming `Struct<Dispatch>` by unwrapping each tensor
        // field into this backend's `BackendTensor`. Inverse of the `map_to_dispatch` output path.
        ArgKind::ExtensionStruct(ty) => {
            let name = &a.name;
            let b_ty = struct_ty_with_param(ty, quote! { #b_ident });
            Some(quote! {
                let #name = <#b_ty as burn::backend::ExtensionType<#b_ident>>::map_from_dispatch(
                    #name,
                    |kind| match kind {
                        burn::backend::DispatchTensorKind::#b_ident(bt) => bt,
                        #[allow(unreachable_patterns)]
                        _ => panic!(
                            "backend extension op received tensor inputs on mismatched backends; all inputs must be on the same backend"
                        ),
                    },
                );
            })
        }
        _ => None,
    });

    let call_args = op.inputs.iter().map(|a| &a.name);
    let maybe_await = if op.asyncness {
        quote! { .await }
    } else {
        quote! {}
    };

    let wrap_out = gen_output_wrap(op, &b_ident, false);

    quote! {
        #(#unwraps)*
        let _out = <#b_ident as #trait_name>::#fn_name(#(#call_args),*)#maybe_await;
        #wrap_out
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

        let wrap_out = gen_output_wrap(op, &b_ident, true);

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
                        #f_name: self.#f_name.map_to_dispatch(
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

    // Reverse of `fields_wrap`: rebuild `Struct<B>` from the incoming `Struct<Dispatch>` (`target`).
    let fields_unwrap = match &input.fields {
        syn::Fields::Named(fields) => {
            let field_mappings = fields.named.iter().map(|f| {
                let f_name = &f.ident;
                let is_ext = f.attrs.iter().any(|attr| attr.path().is_ident("extension_type"));

                if is_ext {
                    // Nested extension type: recurse with the same unwrap closure.
                    quote! {
                        #f_name: burn::backend::ExtensionType::map_from_dispatch(target.#f_name, &unwrap_kind),
                    }
                } else if let Some(tensor_kind) = TensorKind::from_type(&f.ty) {
                    // Tensor primitive: unwrap the DispatchTensor kind, then pull out the concrete
                    // primitive with the accessor matching this field's kind (`.float()`, ...).
                    let method = tensor_kind.unwrap_method();
                    quote! {
                        #f_name: unwrap_kind(target.#f_name.kind).#method(),
                    }
                } else {
                    // Passthrough
                    quote! { #f_name: target.#f_name, }
                }
            });

            quote! { #name { #( #field_mappings )* } }
        }
        _ => unreachable!("named-fields check already happened above"),
    };

    // Representative field: identifies the backend (`.kind`) and carries the checkpointing strategy
    // (`.checkpointing`). Prefer the first float field (the tracked one under autodiff), then any
    // tensor field, then recurse into the first nested `#[extension_type]` field.
    let dispatch_repr_expr = match &input.fields {
        syn::Fields::Named(fields) => {
            let is_plain_tensor = |f: &&syn::Field| {
                !f.attrs
                    .iter()
                    .any(|attr| attr.path().is_ident("extension_type"))
                    && TensorKind::from_type(&f.ty).is_some()
            };
            let first_float = fields.named.iter().find(|f| {
                is_plain_tensor(f) && TensorKind::from_type(&f.ty) == Some(TensorKind::Float)
            });
            let repr = first_float.or_else(|| fields.named.iter().find(is_plain_tensor));
            if let Some(f) = repr {
                let f_name = &f.ident;
                quote! { &target.#f_name }
            } else if let Some(f) = fields.named.iter().find(|f| {
                f.attrs
                    .iter()
                    .any(|attr| attr.path().is_ident("extension_type"))
            }) {
                let f_name = &f.ident;
                quote! { burn::backend::ExtensionType::dispatch_repr(&target.#f_name) }
            } else {
                // No tensor or nested field to represent the struct. This only matters when the
                // struct is used as an *input* (where the dispatch glue calls `dispatch_repr` to
                // select a backend); an output-only struct never calls it. So panic at runtime rather
                // than `compile_error!`, which would otherwise break valid output-only structs.
                quote! {
                    {
                        let _ = target;
                        panic!("an ExtensionType struct used as a backend extension input must contain at least one tensor field to select the backend from")
                    }
                }
            }
        }
        _ => unreachable!("named-fields check already happened above"),
    };

    TokenStream::from(quote! {
        impl #impl_generics burn::backend::ExtensionType<B> for #name #ty_generics #where_clause {
            type Target = #name<burn::backend::Dispatch>;

            fn map_to_dispatch<F>(
                self,
                map_kind: F,
                checkpointing: Option<burn::backend::CheckpointingStrategy>,
            ) -> Self::Target
            where
                F: Fn(burn::backend::BackendTensor<B>) -> burn::backend::DispatchTensorKind,
            {
                #fields_wrap
            }

            fn map_from_dispatch<F>(target: Self::Target, unwrap_kind: F) -> Self
            where
                F: Fn(burn::backend::DispatchTensorKind) -> burn::backend::BackendTensor<B>,
            {
                #fields_unwrap
            }

            fn dispatch_repr(target: &Self::Target) -> &burn::backend::DispatchTensor {
                #dispatch_repr_expr
            }
        }
    })
}
