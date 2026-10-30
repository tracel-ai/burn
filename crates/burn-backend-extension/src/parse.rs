//! The model of an extension trait's lazily executed methods, read once from their
//! `#[fusion(...)]` annotations.
//!
//! `fusion` and `remote` emit from this one model: the `Fusion<B>` implementation, and the remote
//! backend's client and server handlers. Reading the arguments and outputs in one place keeps the
//! IR the two build in one order, and a malformed method is reported once.
use crate::ir::{OperationOutput, TensorKind, with_backend};
use proc_macro2::TokenStream;
use quote::{format_ident, quote, quote_spanned};
use syn::spanned::Spanned;
use syn::{Expr, FnArg, ItemTrait, Pat, ReturnType, TraitItem, TraitItemFn, Type};

/// The lazily executed methods of an extension trait.
pub struct LazyTrait<'a> {
    pub name: &'a syn::Ident,
    pub vis: &'a syn::Visibility,
    /// Every method that does not inherit its default body, in declaration order, or why it
    /// cannot be lowered.
    pub methods: Vec<Result<LazyMethod<'a>, MalformedMethod<'a>>>,
}

impl<'a> LazyTrait<'a> {
    /// Parse every method of `item`.
    pub fn new(item: &'a ItemTrait) -> Self {
        let methods = item
            .items
            .iter()
            .filter_map(|item| match item {
                TraitItem::Fn(method) => Some(method),
                _ => None,
            })
            .filter_map(|method| match LazyMethod::new(&item.ident, method) {
                Ok(lazy) => lazy.map(Ok),
                Err(error) => Some(Err(MalformedMethod::new(method, error))),
            })
            .collect();
        Self {
            name: &item.ident,
            vis: &item.vis,
            methods,
        }
    }
}

/// A method that cannot be lowered, with the error saying why.
///
/// An emitter still implements it, with a body standing in for the lowering: the implementation
/// stays complete, so the error is the only one, and the method's own attributes keep a `cfg`-gated
/// method's error behind its gates.
pub struct MalformedMethod<'a> {
    method: &'a TraitItemFn,
    error: syn::Error,
}

impl<'a> MalformedMethod<'a> {
    pub fn new(method: &'a TraitItemFn, error: syn::Error) -> Self {
        Self { method, error }
    }

    /// The method, whose body reports the error.
    pub fn reported(&self) -> TokenStream {
        self.implementation(self.error.to_compile_error())
    }

    /// The method, for an emitter beside the one reporting the error: a second report would only
    /// repeat it, and the build fails on the first.
    pub fn unreported(&self) -> TokenStream {
        self.implementation(quote!(unreachable!(
            "the method's lowering error is reported once"
        )))
    }

    fn implementation(&self, body: TokenStream) -> TokenStream {
        let attrs = method_attrs(self.method);
        let signature = stripped_signature(self.method);
        quote!(#(#attrs)* #signature { #body })
    }
}

/// A trait method executed lazily from output metadata: the input Fusion and the remote backend
/// generate their implementations from.
pub struct LazyMethod<'a> {
    pub method: &'a TraitItemFn,
    /// The signature without the macro's argument attributes.
    pub signature: syn::Signature,
    pub id: TokenStream,
    metadata: Expr,
    pub arguments: Vec<Argument<'a>>,
    pub output: LazyOutput,
}

impl<'a> LazyMethod<'a> {
    /// Parse a method of `trait_name` from its `#[fusion(...)]` annotations, or `None` when it
    /// inherits its default body.
    fn new(trait_name: &syn::Ident, method: &'a TraitItemFn) -> syn::Result<Option<Self>> {
        let annotations: Vec<_> = method
            .attrs
            .iter()
            .filter(|a| a.path().is_ident("fusion"))
            .collect();
        if annotations.len() != 1 {
            return Err(syn::Error::new_spanned(
                method,
                "each Fusion method requires exactly one #[fusion(dtype = ..., shape = ...)], #[fusion(meta = callable)], or #[fusion(default)]",
            ));
        }
        let (behavior, id) = FusionBehavior::new(annotations[0])?;
        let sig = &method.sig;
        let signature = stripped_signature(method);
        let typed: Vec<_> = sig
            .inputs
            .iter()
            .map(|a| match a {
                FnArg::Typed(p) => match p.pat.as_ref() {
                    Pat::Ident(i) => Ok((&i.ident, p)),
                    _ => Err(unsupported(a)),
                },
                _ => Err(unsupported(a)),
            })
            .collect::<syn::Result<_>>()?;
        let metadata = match behavior {
            FusionBehavior::Default => {
                if id.is_some() {
                    return Err(syn::Error::new_spanned(
                        annotations[0],
                        "default does not register an operation; id does not apply",
                    ));
                }
                if method.default.is_none() {
                    return Err(syn::Error::new_spanned(
                        method,
                        "#[fusion(default)] requires a default body",
                    ));
                }
                return Ok(None);
            }
            FusionBehavior::Meta(expr) => expr,
            FusionBehavior::Tensor { dtype, shape } => {
                if !matches!(&sig.output, ReturnType::Type(_, ty) if TensorKind::from_type(ty).is_some())
                {
                    return Err(syn::Error::new_spanned(
                        &sig.output,
                        "dtype and shape require a single tensor output; use #[fusion(meta = callable)] for structured outputs",
                    ));
                }
                tensor_metadata(dtype, shape, &typed)?
            }
        };
        if !matches!(metadata, Expr::Path(_) | Expr::Closure(_)) {
            return Err(unsupported(metadata));
        }
        if sig.asyncness.is_some() || !sig.generics.params.is_empty() {
            return Err(unsupported(sig));
        }
        let arguments = typed
            .into_iter()
            .map(|(name, arg)| Argument::new(name, arg))
            .collect::<syn::Result<Vec<_>>>()?;
        if !arguments
            .iter()
            .any(|argument| !matches!(argument.kind, ArgumentKind::Ordinary { .. }))
        {
            return Err(unsupported(sig));
        }
        if let Expr::Closure(closure) = &metadata
            && closure.inputs.len() != arguments.len()
        {
            return Err(syn::Error::new_spanned(
                closure,
                "metadata must accept every argument in declaration order",
            ));
        }
        let ReturnType::Type(_, ty) = &sig.output else {
            return Err(unsupported(sig));
        };
        let output = LazyOutput::new(ty)?;
        // The default id names the trait as well as the method: a server registers the handlers
        // of several traits in one registry, and their methods may share a name.
        let id = id.map(|id| quote!(#id)).unwrap_or_else(|| {
            let name = &sig.ident;
            quote!(concat!(stringify!(#trait_name), "::", stringify!(#name)))
        });
        Ok(Some(Self {
            method,
            signature,
            id,
            metadata,
            arguments,
            output,
        }))
    }

    /// The metadata callable. An immediately invoked closure needs explicit parameter types for
    /// expressions like `x.shape`, so untyped parameters get them for `backend`.
    pub fn metadata(&self, backend: TokenStream) -> Expr {
        let mut expr = self.metadata.clone();
        if let Expr::Closure(closure) = &mut expr {
            for (pat, argument) in closure.inputs.iter_mut().zip(&self.arguments) {
                if !matches!(pat, Pat::Type(_)) {
                    let ty = match &argument.kind {
                        ArgumentKind::Tensor { .. } => {
                            quote!(burn::backend::fusion::custom::TensorSpec)
                        }
                        ArgumentKind::Extension { ty } => {
                            let ty = with_backend(ty, backend.clone());
                            quote!(<#ty as burn::backend::fusion::custom::ExtensionMetadata>::Metadata)
                        }
                        ArgumentKind::Ordinary { .. } => {
                            let ty = &argument.arg.ty;
                            quote!(#ty)
                        }
                    };
                    *pat = Pat::Type(syn::PatType {
                        attrs: Vec::new(),
                        pat: Box::new(pat.clone()),
                        colon_token: Default::default(),
                        ty: Box::new(syn::parse_quote!(&#ty)),
                    });
                }
            }
        }
        expr
    }

    /// The method's attributes other than the macro's own.
    pub fn attrs(&self) -> impl Iterator<Item = &syn::Attribute> {
        method_attrs(self.method)
    }

    /// The attributes deciding whether the method exists.
    pub fn gates(&self) -> impl Iterator<Item = &syn::Attribute> {
        self.method
            .attrs
            .iter()
            .filter(|a| a.path().is_ident("cfg") || a.path().is_ident("cfg_attr"))
    }

    /// The scalar values of the ordinary arguments, in declaration order.
    pub fn scalars(&self, check: DecodeCheck) -> Vec<TokenStream> {
        self.arguments
            .iter()
            .filter_map(|argument| match &argument.kind {
                ArgumentKind::Ordinary {
                    scalar: Some(scalar),
                } => Some(scalar.to_ir(check)),
                _ => None,
            })
            .collect()
    }

    /// Statements appending each extension argument's variant and ordinary fields to `__scalars`,
    /// after the ordinary arguments' scalars, read from the `__metadata_{name}` bindings.
    pub fn extension_scalars(&self, backend: TokenStream) -> Vec<TokenStream> {
        self.arguments
            .iter()
            .filter(|argument| matches!(argument.kind, ArgumentKind::Extension { .. }))
            .map(|argument| {
                let adapter = argument.adapter(backend.clone()).expect("extension adapter");
                let meta = format_ident!("__metadata_{}", argument.name);
                quote!(<#adapter as burn::backend::fusion::custom::RemoteValueAdapter<#backend>>::append_scalars(&#meta, &mut __scalars);)
            })
            .collect()
    }

    /// Statements appending the variant of every enum in the output, described by `__meta`, to
    /// `__scalars`, after the inputs' scalars.
    pub fn output_variants(&self, backend: &TokenStream) -> Vec<TokenStream> {
        let mut statements = Vec::new();
        self.output.map_leaves(backend, &mut |adapter, member| {
            statements.push(quote!(<#adapter as burn::backend::fusion::custom::RemoteValueAdapter<#backend>>::append_output_variants(&__meta #member, &mut __scalars);));
            quote!()
        });
        statements
    }

    /// The call of the method on `backend` with the given arguments.
    pub fn call(
        &self,
        trait_name: &syn::Ident,
        backend: TokenStream,
        arguments: &[TokenStream],
    ) -> TokenStream {
        let name = &self.signature.ident;
        let call = quote!(<#backend as #trait_name>::#name(#(#arguments),*));
        if matches!(self.signature.safety, syn::Safety::Unsafe(_)) {
            quote!(unsafe { #call })
        } else {
            call
        }
    }
}

pub struct Argument<'a> {
    pub name: &'a syn::Ident,
    pub arg: &'a syn::PatType,
    pub kind: ArgumentKind<'a>,
}

impl<'a> Argument<'a> {
    /// Classify the argument `name` of a lazily executed method.
    fn new(name: &'a syn::Ident, arg: &'a syn::PatType) -> syn::Result<Self> {
        let scalar = scalar_argument(name, arg)?;
        let is_ext = arg
            .attrs
            .iter()
            .any(|a| a.path().is_ident("extension_type"));
        if matches!(arg.ty.as_ref(), Type::Reference(r) if r.mutability.is_some()) {
            return Err(unsupported(arg));
        }
        let borrowed = matches!(arg.ty.as_ref(), Type::Reference(_));
        let ty = match arg.ty.as_ref() {
            Type::Reference(r) => r.elem.as_ref(),
            ty => ty,
        };
        let kind = if let Some(kind) = TensorKind::from_type(ty).filter(|_| !is_ext) {
            ArgumentKind::Tensor { kind, borrowed }
        } else if is_ext {
            if borrowed {
                return Err(unsupported(arg));
            }
            ArgumentKind::Extension { ty }
        } else {
            if borrowed
                || matches!(arg.ty.as_ref(), Type::ImplTrait(_))
                || crate::ir::type_contains_self(&arg.ty)
            {
                return Err(unsupported(arg));
            }
            ArgumentKind::Ordinary { scalar }
        };
        if !matches!(kind, ArgumentKind::Ordinary { .. })
            && arg.attrs.iter().any(|a| a.path().is_ident("fusion"))
        {
            return Err(syn::Error::new_spanned(
                arg,
                "#[fusion(scalar)] applies to ordinary arguments, not tensor or extension inputs",
            ));
        }
        Ok(Self { name, arg, kind })
    }

    /// The adapter type implementing a runtime support trait for this tensor or extension value.
    pub fn adapter(&self, backend: TokenStream) -> Option<TokenStream> {
        match &self.kind {
            ArgumentKind::Tensor { kind, .. } => Some(tensor_adapter(*kind)),
            ArgumentKind::Extension { ty } => Some(with_backend(ty, backend)),
            ArgumentKind::Ordinary { .. } => None,
        }
    }
}

/// What an argument of a lazily executed method is.
// Parsed once per argument at expansion time, so boxing the scalar encoding buys nothing.
#[allow(clippy::large_enum_variant)]
pub enum ArgumentKind<'a> {
    /// A tensor primitive, owned or borrowed.
    Tensor { kind: TensorKind, borrowed: bool },
    /// A derived extension value, owned.
    Extension { ty: &'a Type },
    /// Any other owned argument, with its scalar encoding when it has one.
    Ordinary { scalar: Option<ScalarArgument> },
}

/// What a lazily executed method returns: tensors, derived extension values, and tuples of those.
pub enum LazyOutput {
    Tensor(TensorKind),
    Extension(Box<Type>),
    Tuple(Vec<LazyOutput>),
}

impl LazyOutput {
    fn new(ty: &Type) -> syn::Result<Self> {
        reject_borrowed_output(ty)?;
        Self::try_from(OperationOutput::extension(ty))
    }

    /// The output's shape with each tensor or extension leaf replaced by `leaf(adapter, member)`,
    /// in leaf order: `adapter` is the leaf's adapter type for `backend`, and `member` the field
    /// access reaching the leaf from the whole output (`.1 .0`, empty for a leaf output).
    pub fn map_leaves(
        &self,
        backend: &TokenStream,
        leaf: &mut impl FnMut(&TokenStream, &TokenStream) -> TokenStream,
    ) -> TokenStream {
        self.map_leaves_at(backend, quote!(), leaf)
    }

    fn map_leaves_at(
        &self,
        backend: &TokenStream,
        member: TokenStream,
        leaf: &mut impl FnMut(&TokenStream, &TokenStream) -> TokenStream,
    ) -> TokenStream {
        match self {
            Self::Tensor(kind) => leaf(&tensor_adapter(*kind), &member),
            Self::Extension(ty) => leaf(&with_backend(ty, backend.clone()), &member),
            Self::Tuple(items) => {
                let items = items.iter().enumerate().map(|(i, item)| {
                    let i = syn::Index::from(i);
                    item.map_leaves_at(backend, quote!(#member.#i), leaf)
                });
                let items: Vec<_> = items.collect();
                quote!((#(#items,)*))
            }
        }
    }
}

impl TryFrom<OperationOutput> for LazyOutput {
    type Error = syn::Error;

    fn try_from(output: OperationOutput) -> syn::Result<Self> {
        match output {
            OperationOutput::Tensor(kind) => Ok(Self::Tensor(kind)),
            OperationOutput::Extension(ty) => Ok(Self::Extension(ty)),
            OperationOutput::Tuple(items) => items
                .into_iter()
                .map(Self::try_from)
                .collect::<syn::Result<_>>()
                .map(Self::Tuple),
            _ => Err(unsupported(quote!(output))),
        }
    }
}

/// How an ordinary argument travels as an IR scalar, and how the receiving side rebuilds it.
pub struct ScalarArgument {
    /// The argument or field the encoding sees.
    name: syn::Ident,
    /// Evaluated with the argument in scope; the result converts into `Scalar`.
    encode: Expr,
    decode: ScalarDecode,
}

impl ScalarArgument {
    /// The encoding of `name`, of type `ty`: its scalar `annotation`, or `name` itself when `ty`
    /// is a primitive; `None` without either.
    pub fn new(
        name: &syn::Ident,
        ty: &Type,
        annotation: Option<&syn::Attribute>,
    ) -> syn::Result<Option<Self>> {
        if let Some(annotation) = annotation {
            return parse_scalar_attribute(annotation, name).map(Some);
        }
        Ok(is_primitive(ty).then(|| Self {
            name: name.clone(),
            encode: syn::parse_quote!(#name),
            decode: ScalarDecode::FromScalar,
        }))
    }

    /// The `ScalarIr` expression of the encoding.
    ///
    /// With [`DecodeCheck::Enforce`] and a decoding callable, the encoded value must have the type
    /// the callable takes, so encoding another primitive is a compile error at the annotation
    /// rather than a scalar the server converts.
    pub fn to_ir(&self, check: DecodeCheck) -> TokenStream {
        let encode = &self.encode;
        match (&self.decode, check) {
            (ScalarDecode::Callable(decode), DecodeCheck::Enforce) => {
                quote_spanned!(decode.span()=> burn::backend::fusion::custom::encode_scalar((#encode).clone(), &(#decode)))
            }
            _ => quote!(burn::backend::fusion::custom::ScalarIr::from(
                burn::backend::Scalar::from((#encode).clone())
            )),
        }
    }

    /// The expression rebuilding a value of type `ty` from `next`, the scalar it travelled as.
    ///
    /// # Errors
    ///
    /// If the encoding is custom and has no decoding callable.
    pub fn decode(&self, ty: &Type, next: TokenStream) -> syn::Result<TokenStream> {
        let support = quote!(burn::backend::fusion::custom);
        match &self.decode {
            ScalarDecode::FromScalar => {
                Ok(quote!(<#ty as #support::FromScalarIr>::from_scalar_ir(#next)))
            }
            ScalarDecode::Callable(decode) => {
                Ok(quote!(#support::decode_scalar::<_, #ty>(#next, #decode)))
            }
            ScalarDecode::Missing => {
                let name = &self.name;
                Err(syn::Error::new_spanned(
                    &self.encode,
                    format!(
                        "the remote backend rebuilds `{name}` from its scalar; add \
                         `decode = <callable>` beside `scalar = ..`, taking the encoded primitive \
                         and returning the value"
                    ),
                ))
            }
        }
    }
}

/// Whether an encoded scalar is tied, at compile time, to the type its decoding callable takes.
#[derive(Clone, Copy)]
pub enum DecodeCheck {
    /// Where only Fusion reads the IR: nothing decodes, and the callable may name items that
    /// exist only where the remote backend does.
    Skip,
    /// Where the remote backend is generated, whose server decodes the scalar.
    Enforce,
}

/// How a scalar is turned back into the argument it encoded.
#[allow(clippy::large_enum_variant)]
enum ScalarDecode {
    /// The argument's type implements `FromScalarIr`: a primitive, or an alias of one.
    FromScalar,
    /// A callable taking the encoded primitive and returning the argument.
    Callable(Expr),
    /// A custom encoding without its decoding, which only the remote backend needs.
    Missing,
}

pub fn unsupported(item: impl quote::ToTokens) -> syn::Error {
    syn::Error::new_spanned(
        item,
        "unsupported lazy Fusion signature; use #[fusion(default)] or omit Fusion from #[backend_extension] and implement the trait manually",
    )
}

/// The method's behavior under Fusion, as its `#[fusion(...)]` annotation selects it.
#[allow(clippy::large_enum_variant)]
enum FusionBehavior {
    Default,
    Meta(Expr),
    Tensor { dtype: Expr, shape: Expr },
}

impl FusionBehavior {
    /// Parse the behavior and the explicit operation id of `annotation`.
    fn new(annotation: &syn::Attribute) -> syn::Result<(Self, Option<syn::LitStr>)> {
        let mut behavior = None;
        let mut dtype = None;
        let mut shape = None;
        let mut id = None;
        annotation.parse_nested_meta(|meta| {
            if meta.path.is_ident("id") {
                if id.is_some() {
                    return Err(meta.error("duplicate Fusion id"));
                }
                id = Some(meta.value()?.parse::<syn::LitStr>()?);
                return Ok(());
            }
            if meta.path.is_ident("dtype") || meta.path.is_ident("shape") {
                if behavior.is_some() {
                    return Err(
                        meta.error("dtype and shape cannot be combined with meta or default")
                    );
                }
                let slot = if meta.path.is_ident("dtype") {
                    &mut dtype
                } else {
                    &mut shape
                };
                if slot.is_some() {
                    return Err(meta.error("duplicate Fusion metadata field"));
                }
                *slot = Some(meta.value()?.parse::<Expr>()?);
                return Ok(());
            }
            if behavior.is_some() || dtype.is_some() || shape.is_some() {
                return Err(meta.error("conflicting Fusion behaviors"));
            }
            behavior = Some(if meta.path.is_ident("default") {
                Self::Default
            } else if meta.path.is_ident("meta") {
                Self::Meta(meta.value()?.parse()?)
            } else {
                return Err(meta.error("expected dtype, shape, meta, default, or id"));
            });
            Ok(())
        })?;
        let behavior = match (behavior, dtype, shape) {
            (Some(behavior), None, None) => behavior,
            (None, Some(dtype), Some(shape)) => Self::Tensor { dtype, shape },
            (None, _, _) => {
                return Err(syn::Error::new_spanned(
                    annotation,
                    "provide both dtype and shape, or use meta or default",
                ));
            }
            _ => unreachable!("conflicting behaviors are rejected while parsing"),
        };
        Ok((behavior, id))
    }
}

/// The method's attributes other than the macro's own.
fn method_attrs(method: &TraitItemFn) -> impl Iterator<Item = &syn::Attribute> {
    method.attrs.iter().filter(|a| !a.path().is_ident("fusion"))
}

/// The method's signature without the macro's argument attributes.
fn stripped_signature(method: &TraitItemFn) -> syn::Signature {
    let mut signature = method.sig.clone();
    strip_argument_attributes(&mut signature);
    signature
}

/// Remove the macro's argument attributes, `#[extension_type]` and `#[fusion(..)]`, which rustc
/// rejects as unknown once the macro has read them.
pub fn strip_argument_attributes(signature: &mut syn::Signature) {
    for arg in &mut signature.inputs {
        if let FnArg::Typed(arg) = arg {
            arg.attrs
                .retain(|a| !a.path().is_ident("extension_type") && !a.path().is_ident("fusion"));
        }
    }
}

fn tensor_adapter(kind: TensorKind) -> TokenStream {
    let kind = kind.variant();
    quote!(burn::backend::fusion::custom::#kind)
}

/// Primitive parameters are scalar IR inputs; custom types can opt in or provide an encoding.
/// The macro cannot resolve aliases or inspect a type declared in another crate.
fn scalar_argument(name: &syn::Ident, arg: &syn::PatType) -> syn::Result<Option<ScalarArgument>> {
    let attrs: Vec<_> = arg
        .attrs
        .iter()
        .filter(|a| a.path().is_ident("fusion"))
        .collect();
    if attrs.len() > 1 {
        return Err(syn::Error::new_spanned(
            arg,
            "expected one #[fusion(scalar)] or #[fusion(scalar = expression)]",
        ));
    }
    ScalarArgument::new(name, &arg.ty, attrs.first().copied())
}

/// Parse `scalar`, `scalar = expression`, and an optional `decode = callable` from one attribute.
/// A bare `scalar` encodes `name` itself.
fn parse_scalar_attribute(attr: &syn::Attribute, name: &syn::Ident) -> syn::Result<ScalarArgument> {
    let mut encode: Option<Option<Expr>> = None;
    let mut decode = None;
    attr.parse_nested_meta(|meta| {
        if meta.path.is_ident("scalar") && encode.is_none() {
            encode = Some(if meta.input.peek(syn::Token![=]) {
                Some(meta.value()?.parse()?)
            } else {
                None
            });
            return Ok(());
        }
        if meta.path.is_ident("decode") && decode.is_none() {
            decode = Some(meta.value()?.parse::<Expr>()?);
            return Ok(());
        }
        Err(meta.error("expected scalar, scalar = expression, or decode = callable"))
    })?;
    let Some(encode) = encode else {
        return Err(syn::Error::new_spanned(
            attr,
            "expected scalar or scalar = expression",
        ));
    };
    let decode = match (&encode, decode) {
        (_, Some(decode)) => ScalarDecode::Callable(decode),
        (None, None) => ScalarDecode::FromScalar,
        (Some(_), None) => ScalarDecode::Missing,
    };
    Ok(ScalarArgument {
        name: name.clone(),
        encode: encode.unwrap_or_else(|| syn::parse_quote!(#name)),
        decode,
    })
}

/// Whether `ty` names a primitive that converts into `Scalar` and back.
fn is_primitive(ty: &Type) -> bool {
    let Type::Path(ty) = ty else {
        return false;
    };
    let segments: Vec<_> = ty.path.segments.iter().collect();
    let primitive_path = segments.len() == 1
        || (segments.len() == 3
            && (segments[0].ident == "core" || segments[0].ident == "std")
            && segments[1].ident == "primitive");
    ty.qself.is_none()
        && primitive_path
        && segments
            .iter()
            .all(|s| matches!(s.arguments, syn::PathArguments::None))
        && segments.last().is_some_and(|s| {
            matches!(
                s.ident.to_string().as_str(),
                "bool"
                    | "f32"
                    | "f64"
                    | "i8"
                    | "i16"
                    | "i32"
                    | "i64"
                    | "isize"
                    | "u8"
                    | "u16"
                    | "u32"
                    | "u64"
                    | "usize"
            )
        })
}

/// Lower field expressions to the same metadata callback used by `meta`.
/// Separate scopes give tensor names their dtype or borrowed shape without rewriting expressions.
fn tensor_metadata(
    dtype: Expr,
    shape: Expr,
    args: &[(&syn::Ident, &syn::PatType)],
) -> syn::Result<Expr> {
    let names: Vec<_> = args.iter().map(|(name, _)| name).collect();
    let tensors: Vec<_> = args
        .iter()
        .filter(|(_, arg)| TensorKind::from_type(&arg.ty).is_some())
        .map(|(name, _)| *name)
        .collect();
    // A bare operand borrows its shape; computed expressions already return an owned Shape.
    let shape = if matches!(&shape, Expr::Path(path) if tensors.iter().any(|name| path.path.is_ident(*name)))
    {
        quote!((#shape).clone())
    } else {
        quote!(#shape)
    };
    syn::parse2(quote! {
        |#(#names),*| {
            let _ = (#(#names),*);
            let __dtype = { #(#[allow(unused_variables)] let #tensors = #tensors.dtype;)* #dtype };
            let __shape = { #(#[allow(unused_variables)] let #tensors = &#tensors.shape;)* #shape };
            burn::backend::fusion::custom::TensorSpec::new(__shape, __dtype)
        }
    })
}

fn reject_borrowed_output(ty: &Type) -> syn::Result<()> {
    match ty {
        Type::Reference(_) => Err(unsupported(ty)),
        Type::Tuple(tuple) => {
            for ty in &tuple.elems {
                reject_borrowed_output(ty)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_parameter_annotations_are_validated() {
        for argument in [
            quote!(#[fusion()] value: f32),
            quote!(#[fusion(other)] value: f32),
            quote!(#[fusion(scalar, scalar)] value: f32),
            quote!(#[fusion(scalar)] #[fusion(scalar)] value: f32),
            quote!(#[fusion(scalar)] value: FloatTensor<Self>),
            quote!(#[fusion(scalar)] #[extension_type] value: Options<Self>),
        ] {
            let error = parse_error(quote! {
                trait Ext: Backend {
                    #[fusion(dtype = x, shape = x)]
                    fn op(x: FloatTensor<Self>, #argument) -> FloatTensor<Self>;
                }
            });
            assert!(error.is_some(), "accepted {argument}");
        }
    }

    #[test]
    fn rejects_incomplete_duplicate_and_conflicting_fields() {
        for attr in [
            quote!(dtype = x),
            quote!(shape = x),
            quote!(dtype = x, dtype = x, shape = x),
            quote!(shape = x, shape = x, dtype = x),
            quote!(meta = metadata, dtype = x, shape = x),
            quote!(dtype = x, shape = x, meta = metadata),
            quote!(default, dtype = x, shape = x),
            quote!(shape = x, default),
            quote!(default, id = "op"),
            quote!(default, scalars = []),
            quote!(dtype = x, shape = x, id = "a", id = "b"),
            quote!(dtype = x, shape = x, scalars = [], scalars = []),
        ] {
            let error = parse_error(quote! {
                trait Ext: Backend {
                    #[fusion(#attr)]
                    fn op(x: FloatTensor<Self>) -> FloatTensor<Self> { x }
                }
            });
            assert!(error.is_some(), "accepted {attr}");
        }
        let error = parse_error(quote! {
            trait Ext: Backend {
                #[fusion(dtype = x, shape = x)]
                fn op(x: FloatTensor<Self>) -> (FloatTensor<Self>, FloatTensor<Self>);
            }
        });
        assert!(error.unwrap().contains("single tensor output"));
    }
    #[test]
    fn rejects_missing_conflicting_and_unsupported_behaviors() {
        for method in [
            quote!(
                #[fusion(custom = helper)]
                fn op(x: FloatTensor<Self>) -> FloatTensor<Self>;
            ),
            quote!(
                fn op(x: FloatTensor<Self>) -> FloatTensor<Self>;
            ),
            quote!(
                #[fusion(default)]
                fn op(x: FloatTensor<Self>) -> FloatTensor<Self>;
            ),
            quote!(
                #[fusion(meta = meta, default)]
                fn op(x: FloatTensor<Self>) -> FloatTensor<Self>;
            ),
            quote!(
                #[fusion(meta = meta)]
                async fn op(x: FloatTensor<Self>) -> FloatTensor<Self>;
            ),
            quote!(
                #[fusion(meta = meta)]
                fn op<T>(x: FloatTensor<Self>) -> FloatTensor<Self>;
            ),
            quote!(
                #[fusion(meta = meta)]
                fn op(x: FloatTensor<Self>, v: &usize) -> FloatTensor<Self>;
            ),
            quote!(
                #[fusion(dtype = x, shape = x)]
                fn op(x: &mut FloatTensor<Self>) -> FloatTensor<Self>;
            ),
            quote!(
                #[fusion(meta = meta)]
                fn op(x: FloatTensor<Self>) -> Vec<FloatTensor<Self>>;
            ),
            quote!(
                #[fusion(meta = meta)]
                fn op(x: usize) -> FloatTensor<Self>;
            ),
        ] {
            assert!(
                parse_error(quote!(trait Ext: Backend { #method })).is_some(),
                "accepted {method}"
            );
        }
    }
    #[test]
    fn detects_primitive_scalars_and_requires_alias_opt_in() {
        for (ty, expected) in [
            (quote!(f32), true),
            (quote!(f64), true),
            (quote!(bool), true),
            (quote!(usize), true),
            (quote!(isize), true),
            (quote!(u64), true),
            (quote!(core::primitive::i16), true),
            (quote!(std::primitive::u8), true),
            (quote!(Amount), false),
            (quote!(options::u32), false),
            (quote!(String), false),
            (quote!(i128), false),
        ] {
            let arg = syn::parse2(quote!(value: #ty)).unwrap();
            assert_eq!(
                scalar_argument(&syn::parse_quote!(value), &arg)
                    .unwrap()
                    .is_some(),
                expected,
                "{ty}"
            );
        }
    }

    /// A malformed method is still implemented, with its error as the body and its attributes
    /// kept: the implementation stays complete, and a gated method's error stays behind its gate.
    #[test]
    fn a_malformed_method_keeps_its_gates_and_reports_in_its_body() {
        let item = syn::parse2(quote! {
            trait Ext: Backend {
                #[cfg_attr(feature = "a", cfg(feature = "b"))]
                fn op(#[extension_type] x: Operand<Self>) -> FloatTensor<Self>;
            }
        })
        .unwrap();
        let model = LazyTrait::new(&item);
        let Err(malformed) = &model.methods[0] else {
            panic!("accepted a method without an annotation");
        };
        let reported = malformed.reported().to_string();
        assert!(
            reported.starts_with(
                "# [cfg_attr (feature = \"a\" , cfg (feature = \"b\"))] fn op (x : Operand < Self >)"
            ),
            "{reported}"
        );
        assert!(reported.contains("compile_error"), "{reported}");
        assert!(!malformed.unreported().to_string().contains("compile_error"));
    }

    /// The error of the first method that cannot be lowered, if any.
    fn parse_error(item: TokenStream) -> Option<String> {
        let item = syn::parse2(item).unwrap();
        LazyTrait::new(&item)
            .methods
            .into_iter()
            .find_map(|method| method.err().map(|malformed| malformed.error.to_string()))
    }
}
