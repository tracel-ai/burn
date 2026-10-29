//! Emit, from an extension trait's parsed [`LazyTrait`], custom operations that travel as IR alone.
//!
//! The remote backend runs an operation on a server that has nothing but its `CustomOpIr`, so the
//! IR carries every argument: tensors as inputs in declaration order (leaf order inside extension
//! values), outputs from the method's Fusion metadata, and as scalars first every ordinary argument
//! in declaration order, then each extension argument's variant and ordinary fields, then the
//! variant of each enum in the output. The Fusion wrapper builds the same IR, so an operation fused
//! into a cached graph reaches the server the same way as one sent alone.
//!
//! Two items are generated: the client, an implementation for `BackendRouter<R>` that the
//! generated `Fusion<B>` implementation wraps, and the server's handlers, a type registering one
//! handler per method in a `CustomOpRegistry<B>` for any backend `B` implementing the trait. A
//! handler checks the backend's outputs against the IR alone; the server never runs the metadata
//! callable.
use crate::derive::{DeriveCase, gen_case_ctor, gen_case_pattern};
use crate::parse::{
    ArgumentKind, DecodeCheck, LazyMethod, LazyTrait, MalformedMethod, ScalarArgument,
};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

/// The router backend the client is implemented for. Generated generic parameters carry a `__`
/// prefix so they cannot shadow a type the trait's signatures name.
fn router() -> TokenStream {
    quote!(burn::backend::remote_extension::BackendRouter<__R>)
}

/// The client implementation and the server handlers of `model`, both behind `gate`.
///
/// The client reports the error of a method whose arguments the server cannot rebuild; the error
/// of a method that cannot be lowered at all is the Fusion implementation's to report.
pub fn expand(model: &LazyTrait<'_>, gate: &TokenStream) -> TokenStream {
    let trait_name = model.name;
    let mut methods = Vec::new();
    let mut registrations = Vec::new();
    for method in &model.methods {
        match method.as_ref().map(RemoteMethod::new) {
            Ok(Ok(method)) => {
                methods.push(method.client());
                registrations.push(method.handler(trait_name));
            }
            Ok(Err(malformed)) => methods.push(malformed.reported()),
            Err(malformed) => methods.push(malformed.unreported()),
        }
    }
    let vis = model.vis;
    let handlers = format_ident!("{trait_name}Handlers");
    let doc = format!(
        "The server side of [`{trait_name}`] on the remote backend: one custom operation handler \
         per method.\n\nRegister them with the registry the server is given for the backend it \
         hosts, once, before serving."
    );
    quote! {
        #gate
        impl<__R: burn::backend::remote_extension::RouterChannel> #trait_name for burn::backend::remote_extension::BackendRouter<__R> {
            #(#methods)*
        }

        #gate
        #[doc = #doc]
        #vis struct #handlers;

        #gate
        impl #handlers {
            /// Add a handler for every method to `registry`: each rebuilds the call from the
            /// operation's IR, runs it on the backend, and registers the outputs under the IR's
            /// ids. Registering the same trait again is harmless.
            ///
            /// # Panics
            ///
            /// If another handler already claims one of the methods' operation ids, as two
            /// methods given the same `#[fusion(id = ..)]` do.
            pub fn register<__B>(registry: &mut burn::backend::remote_extension::CustomOpRegistry<__B>)
            where
                __B: burn::backend::fusion::custom::BackendIr + #trait_name,
            {
                #(#registrations)*
            }
        }
    }
}

/// A lazily executed method as the remote backend sends and serves it.
struct RemoteMethod<'a> {
    method: &'a LazyMethod<'a>,
    /// The statement rebuilding each ordinary argument from its scalar, in declaration order.
    decodes: Vec<TokenStream>,
}

impl<'a> RemoteMethod<'a> {
    /// # Errors
    ///
    /// An ordinary argument the server cannot rebuild from its scalar.
    fn new(method: &'a LazyMethod<'a>) -> Result<Self, MalformedMethod<'a>> {
        let malformed = |error| MalformedMethod::new(method.method, error);
        let mut decodes = Vec::new();
        for argument in &method.arguments {
            let ArgumentKind::Ordinary { scalar } = &argument.kind else {
                continue;
            };
            let name = argument.name;
            let ty = &argument.arg.ty;
            let Some(scalar) = scalar else {
                return Err(malformed(syn::Error::new_spanned(
                    argument.arg,
                    format!(
                        "the remote backend receives `{name}` as an IR scalar, and this type has no \
                         scalar encoding; annotate it with \
                         `#[fusion(scalar = <expression>, decode = <callable>)]`"
                    ),
                )));
            };
            let next = quote!(burn::backend::fusion::custom::next_scalar(&mut __scalars));
            let decoded = scalar.decode(ty, next).map_err(malformed)?;
            decodes.push(quote!(let #name: #ty = #decoded;));
        }
        Ok(Self { method, decodes })
    }

    /// The `BackendRouter<R>` method: describe the call as IR and send it through the router
    /// client.
    fn client(&self) -> TokenStream {
        let method = self.method;
        let backend = router();
        let support = quote!(burn::backend::fusion::custom);
        let mut metadata_inputs = Vec::new();
        let mut metadata_args = Vec::new();
        let mut inputs = Vec::new();
        for argument in &method.arguments {
            let name = argument.name;
            match argument.adapter(backend.clone()) {
                Some(adapter) => {
                    let adapter = quote!(<#adapter as #support::RemoteValueAdapter<#backend>>);
                    let borrowed =
                        matches!(argument.kind, ArgumentKind::Tensor { borrowed: true, .. });
                    let meta = format_ident!("__metadata_{name}");
                    let (reference, value) = if borrowed {
                        (quote!(#name), quote!((*#name).clone()))
                    } else {
                        (quote!(&#name), quote!(#name))
                    };
                    metadata_inputs.push(quote!(let #meta = #adapter::to_metadata(#reference);));
                    metadata_args.push(quote!(&#meta));
                    inputs.push(quote!(#adapter::append_input(#value, &mut __input);));
                }
                None => metadata_args.push(quote!(&#name)),
            }
        }
        let scalars = method.scalars(DecodeCheck::Enforce);
        let extension_scalars = method.extension_scalars(backend.clone());
        let output_variants = method.output_variants(&backend);
        let mut output_ir = Vec::new();
        let reconstruct = method.output.map_leaves(&backend, &mut |adapter, member| {
            let adapter = quote!(<#adapter as #support::RemoteValueAdapter<#backend>>);
            output_ir.push(
                quote!(#adapter::append_output_ir(&__meta #member, &mut __outputs, &mut __create);),
            );
            quote!(#adapter::build_output(&__meta #member, &mut __handles))
        });
        let expr = method.metadata(backend.clone());
        let id = &method.id;
        let signature = &method.signature;
        let attrs = method.attrs();
        quote! {
            #(#attrs)* #signature {
                use burn::backend::remote_extension::RouterClient as _;
                #(#metadata_inputs)*
                let __meta = (#expr)(#(#metadata_args),*);
                #[allow(unused_mut)]
                let mut __scalars = vec![#(#scalars),*];
                #(#extension_scalars)*
                #(#output_variants)*
                let mut __client = None;
                #[cfg(debug_assertions)]
                let mut __device_id = None;
                let mut __inputs = Vec::new();
                {
                    let mut __input = |tensor: burn::backend::remote_extension::RouterTensor<<__R as burn::backend::remote_extension::RouterChannel>::Client>| {
                        #[cfg(debug_assertions)] {
                            let id = burn::backend::Device::to_id(&tensor.client.device());
                            assert_eq!(*__device_id.get_or_insert(id), id, "custom operation inputs must share a device");
                        }
                        __client.get_or_insert_with(|| tensor.client.clone());
                        __inputs.push(tensor.into_ir());
                    };
                    #(#inputs)*
                }
                let __client = __client.expect("a custom operation requires at least one input tensor");
                let mut __outputs = Vec::new();
                {
                    let mut __create = || __client.create_empty_handle();
                    #(#output_ir)*
                }
                let __desc = #support::CustomOpIr {
                    id: (#id).into(),
                    inputs: __inputs,
                    outputs: __outputs,
                    scalars: __scalars,
                };
                let mut __handles = __client.register(#support::OperationIr::Custom(__desc)).into_iter();
                #reconstruct
            }
        }
    }

    /// The registration of the server handler rebuilding a call of the method on `__B`.
    ///
    /// The handler checks the input and scalar counts and each output tensor's dtype and each
    /// enum's variant against the IR, in every build. On a mismatch it marks every output of the
    /// operation failed, so the client's next read of one reports it, as a failed kernel would.
    /// An IR with fewer inputs or scalars than the signature reads still panics: the call cannot
    /// be rebuilt from it, and neither can a scalar of another kind or one `decode` rejects.
    fn handler(&self, trait_name: &syn::Ident) -> TokenStream {
        let method = self.method;
        let decodes = &self.decodes;
        let mut reads = Vec::new();
        let mut invoke = Vec::new();
        for argument in &method.arguments {
            let name = argument.name;
            match &argument.kind {
                ArgumentKind::Ordinary { .. } => invoke.push(quote!(#name)),
                ArgumentKind::Tensor { borrowed, .. } => {
                    invoke.push(if *borrowed {
                        quote!(&#name)
                    } else {
                        quote!(#name)
                    });
                }
                ArgumentKind::Extension { .. } => invoke.push(quote!(#name)),
            }
            if let Some(adapter) = argument.adapter(quote!(__B)) {
                reads.push(quote! {
                    let #name = <#adapter as burn::backend::fusion::custom::RemoteValueAdapter<__B>>::read_input(__handles, &mut __inputs, &mut __scalars);
                });
            }
        }
        let support = quote!(burn::backend::fusion::custom);
        let mut register = Vec::new();
        method.output.map_leaves(&quote!(__B), &mut |adapter, member| {
            register.push(quote! {
                if let Err(mismatch) = <#adapter as #support::RemoteValueAdapter<__B>>::register_output(__output #member, &mut __outputs, &mut __scalars, __handles) {
                    return mismatch.fail_outputs(__ir, __handles);
                }
            });
            quote!()
        });
        let call = method.call(trait_name, quote!(__B), &invoke);
        let id = &method.id;
        let name = method.method.sig.ident.to_string();
        let gates = method.gates();
        quote! {
            #(#gates)*
            registry.try_register(#id, burn::backend::remote_extension::HandlerSource::new::<Self>(#name), |__handles, __ir, _| {
                let mut __inputs = __ir.inputs.iter();
                let mut __scalars = __ir.scalars.iter();
                #(#decodes)*
                #(#reads)*
                if __inputs.next().is_some() {
                    return #support::IrMismatch::ExtraInput.fail_outputs(__ir, __handles);
                }
                let __output = #call;
                let mut __outputs = __ir.outputs.iter();
                #(#register)*
                if __outputs.next().is_some() {
                    return #support::IrMismatch::MissingTensor.fail_outputs(__ir, __handles);
                }
                if __scalars.next().is_some() {
                    #support::IrMismatch::ExtraScalar.fail_outputs(__ir, __handles);
                }
            })
            .unwrap_or_else(|conflict| {
                panic!("{conflict}; give one of the methods sharing it its own `#[fusion(id = ..)]`")
            });
        }
    }
}

/// The `RemoteValueAdapter` implementation of a derived extension value.
///
/// It is fed by the field walk that builds the value's `FusionValueAdapter`, so the remote and the
/// Fusion adapters visit tensor leaves and ordinary fields in one order: the Fusion wrapper builds
/// a custom operation's inputs and outputs with the one, and the server reads them with the other.
pub struct RemoteAdapterDerive {
    /// The value's type, which names it in a variant mismatch.
    name: syn::Ident,
    backend: syn::Ident,
    is_enum: bool,
    /// The statements of the case being walked.
    case: CaseStatements,
    /// The match arms of each method, one per finished case.
    describe: Vec<TokenStream>,
    encode: Vec<TokenStream>,
    append: Vec<TokenStream>,
    outputs: Vec<TokenStream>,
    variants: Vec<TokenStream>,
    build: Vec<TokenStream>,
    read: Vec<TokenStream>,
    register: Vec<TokenStream>,
}

/// One case's statements and field expressions for each `RemoteValueAdapter` method, in field order.
#[derive(Default)]
struct CaseStatements {
    descriptions: Vec<TokenStream>,
    scalars: Vec<TokenStream>,
    inputs: Vec<TokenStream>,
    output_ir: Vec<TokenStream>,
    variants: Vec<TokenStream>,
    built: Vec<TokenStream>,
    reads: Vec<TokenStream>,
    registrations: Vec<TokenStream>,
}

impl RemoteAdapterDerive {
    /// Start the implementation for the value `name`, generic over `backend`.
    pub fn new(name: &syn::Ident, backend: &syn::Ident, is_enum: bool) -> Self {
        Self {
            name: name.clone(),
            backend: backend.clone(),
            is_enum,
            case: CaseStatements::default(),
            describe: Vec::new(),
            encode: Vec::new(),
            append: Vec::new(),
            outputs: Vec::new(),
            variants: Vec::new(),
            build: Vec::new(),
            read: Vec::new(),
            register: Vec::new(),
        }
    }

    /// Add a tensor or nested extension field, delegating to its own `adapter`. `value` binds the
    /// field in the value and `state` in its metadata.
    pub fn adapted_field(&mut self, adapter: &TokenStream, value: &syn::Ident, state: &syn::Ident) {
        let b = &self.backend;
        let adapter = quote!(<#adapter as burn::backend::fusion::custom::RemoteValueAdapter<#b>>);
        let case = &mut self.case;
        case.descriptions
            .push(quote!(#adapter::to_metadata(#value)));
        case.scalars
            .push(quote!(#adapter::append_scalars(#state, __scalars);));
        case.inputs
            .push(quote!(#adapter::append_input(#value, input);));
        case.output_ir
            .push(quote!(#adapter::append_output_ir(#state, out, create);));
        case.variants
            .push(quote!(#adapter::append_output_variants(#state, __scalars);));
        case.built
            .push(quote!(#adapter::build_output(#state, handles)));
        case.reads
            .push(quote!(#adapter::read_input(handles, inputs, __scalars)));
        case.registrations
            .push(quote!(#adapter::register_output(#value, outputs, variants, handles)?;));
    }

    /// Add an ordinary field, which travels as a scalar.
    pub fn ordinary_field(
        &mut self,
        field: &crate::derive::CaseField,
        value: &syn::Ident,
        state: &syn::Ident,
    ) -> syn::Result<()> {
        let support = quote!(burn::backend::fusion::custom);
        let (name, scalar) = field_scalar(field)?;
        let encoded = scalar.to_ir(DecodeCheck::Enforce);
        let read = scalar.decode(&field.ty, quote!(#support::next_scalar(__scalars)))?;
        let case = &mut self.case;
        case.scalars.push(quote! {{
            #[allow(unused_variables)]
            let #name = #state;
            __scalars.push(#encoded);
        }});
        case.descriptions.push(quote!(#value.clone()));
        case.built.push(quote!(#state.clone()));
        case.reads.push(read);
        Ok(())
    }

    /// Close the case whose fields were just added: `case` is the value's and `meta_case` its
    /// metadata's, whose bindings are distinct.
    pub fn end_case(&mut self, case: &DeriveCase, meta_case: &DeriveCase) {
        let support = quote!(burn::backend::fusion::custom);
        let CaseStatements {
            descriptions,
            scalars,
            inputs,
            output_ir,
            variants,
            built,
            reads,
            registrations,
        } = core::mem::take(&mut self.case);
        let index = self.describe.len() as u64;
        let is_tensor = |i: usize| case.fields[i].tensor_kind.is_some() || case.fields[i].is_ext;
        let value_pattern = gen_case_pattern(case, |_| true);
        let tensor_pattern = gen_case_pattern(case, is_tensor);
        let meta_pattern = gen_case_pattern(meta_case, |_| true);
        let meta_tensor_pattern = gen_case_pattern(meta_case, is_tensor);
        let variant = self
            .is_enum
            .then(|| quote!(__scalars.push(#support::ScalarIr::UInt(#index));));
        let name = &self.name;
        let check_variant = self
            .is_enum
            .then(|| quote!(#support::check_output_variant(variants, stringify!(#name), #index)?;));
        let description = gen_case_ctor(meta_case, &descriptions);
        let rebuilt = gen_case_ctor(case, &built);
        let decoded = gen_case_ctor(case, &reads);
        self.describe.push(quote!(#value_pattern => #description,));
        self.encode
            .push(quote!(#meta_pattern => { #variant #(#scalars)* }));
        self.append.push(quote!(#tensor_pattern => { #(#inputs)* }));
        self.outputs
            .push(quote!(#meta_tensor_pattern => { #(#output_ir)* }));
        self.variants
            .push(quote!(#meta_tensor_pattern => { #variant #(#variants)* }));
        self.build.push(quote!(#meta_pattern => #rebuilt,));
        self.read.push(if self.is_enum {
            quote!(#index => #decoded,)
        } else {
            quote!(_ => #decoded,)
        });
        self.register
            .push(quote!(#tensor_pattern => { #check_variant #(#registrations)* Ok(()) }));
    }

    /// The implementation, for `input` with metadata type `metadata`.
    pub fn finish(
        self,
        input: &syn::DeriveInput,
        backend: &syn::TypeParam,
        metadata: &syn::Ident,
        gate: &Option<TokenStream>,
    ) -> TokenStream {
        let Self {
            backend: b,
            is_enum,
            describe,
            encode,
            append,
            outputs,
            variants,
            build,
            read,
            register,
            ..
        } = self;
        let support = quote!(burn::backend::fusion::custom);
        let name = &input.ident;
        let read = if is_enum {
            quote! {
                match <u64 as #support::FromScalarIr>::from_scalar_ir(#support::next_scalar(__scalars)) {
                    #(#read)*
                    variant => panic!("`{}` has no variant {variant}", stringify!(#name)),
                }
            }
        } else {
            quote!(match () { #(#read)* })
        };
        let bounds = &backend.bounds;
        let where_clause = &input.generics.where_clause;
        quote! {
            #gate
            #[allow(unused_variables)]
            impl<#b: #support::BackendIr + #bounds> #support::RemoteValueAdapter<#b> for #name<#b> #where_clause {
                type Metadata = #metadata;
                type Value = Self;
                fn to_metadata(value: &Self) -> Self::Metadata { match value { #(#describe)* } }
                fn append_scalars(meta: &Self::Metadata, __scalars: &mut Vec<#support::ScalarIr>) { match meta { #(#encode)* } }
                fn append_input(value: Self, input: &mut impl FnMut(#b::Handle)) { match value { #(#append)* } }
                fn append_output_ir(meta: &Self::Metadata, out: &mut Vec<#support::TensorIr>, create: &mut impl FnMut() -> #support::TensorId) { match meta { #(#outputs)* } }
                fn append_output_variants(meta: &Self::Metadata, __scalars: &mut Vec<#support::ScalarIr>) { match meta { #(#variants)* } }
                fn build_output(meta: &Self::Metadata, handles: &mut impl Iterator<Item = #b::Handle>) -> Self { match meta { #(#build)* } }
                fn read_input(
                    handles: &mut #support::HandleContainer<#b::Handle>,
                    inputs: &mut core::slice::Iter<'_, #support::TensorIr>,
                    __scalars: &mut core::slice::Iter<'_, #support::ScalarIr>,
                ) -> Self { #read }
                fn register_output(
                    value: Self,
                    outputs: &mut core::slice::Iter<'_, #support::TensorIr>,
                    variants: &mut core::slice::Iter<'_, #support::ScalarIr>,
                    handles: &mut #support::HandleContainer<#b::Handle>,
                ) -> Result<(), #support::IrMismatch> { match value { #(#register)* } }
            }
        }
    }
}

/// The scalar encoding of an ordinary field: its `#[extension_type(scalar ..)]` annotation, or
/// the field itself when it is a primitive. The encoding sees a named field by its name and a
/// positional one as `value`.
fn field_scalar(field: &crate::derive::CaseField) -> syn::Result<(syn::Ident, ScalarArgument)> {
    let name = field
        .member
        .clone()
        .unwrap_or_else(|| format_ident!("value"));
    match ScalarArgument::new(&name, &field.ty, field.scalar_annotation())? {
        Some(scalar) => Ok((name, scalar)),
        None => Err(syn::Error::new_spanned(
            &field.ty,
            format!(
                "a `remote` extension value sends `{name}` as an IR scalar, and this type has no \
                 scalar encoding; annotate the field with \
                 `#[extension_type(scalar = <expression>, decode = <callable>)]`"
            ),
        )),
    }
}

#[cfg(test)]
mod tests {
    use quote::quote;

    fn expand(backends: proc_macro2::TokenStream, item: proc_macro2::TokenStream) -> String {
        crate::extension::expand(backends, item)
            .unwrap()
            .to_string()
    }

    #[test]
    fn remote_beside_fusion_generates_client_and_handlers() {
        let out = expand(
            quote!(Cube, Fusion: cfg(feature = "fusion"), Remote: cfg(feature = "remote")),
            quote! {
                pub trait Ext: Backend {
                    #[fusion(dtype = x, shape = x)]
                    fn op(
                        x: FloatTensor<Self>,
                        #[fusion(scalar = mode.to_code(), decode = |code| Mode::from_code(code).unwrap())] mode: Mode,
                        amount: usize,
                        #[extension_type] operand: Operand<Self>,
                    ) -> FloatTensor<Self>;
                }
            },
        );
        assert!(out.contains("for burn :: backend :: remote_extension :: BackendRouter < __R >"));
        assert!(out.contains("pub struct ExtHandlers"));
        assert!(out.contains("decode_scalar"));
        // The Fusion wrapper appends the extension's scalars under the remote gate.
        assert!(out.contains("RemoteValueAdapter < B >> :: append_scalars"));
        // The client and the dispatch arm exist only where both Remote and Fusion do.
        assert!(out.contains("# [cfg (feature = \"remote\")] # [cfg (feature = \"fusion\")]"));
        syn::parse_str::<syn::File>(&out).unwrap();
    }

    #[test]
    fn handler_registration_carries_the_method_gates() {
        let out = expand(
            quote!(Flex, Fusion, Remote),
            quote! {
                pub trait Ext: Backend {
                    #[cfg_attr(feature = "a", cfg(feature = "b"))]
                    #[fusion(dtype = x, shape = x)]
                    fn op(x: FloatTensor<Self>) -> FloatTensor<Self>;
                }
            },
        );
        assert!(out.contains(
            "# [cfg_attr (feature = \"a\" , cfg (feature = \"b\"))] registry . try_register"
        ));
    }

    /// A method that cannot be lowered is reported once, by the Fusion implementation; the client
    /// still implements it, so the remote dispatch arm adds no error of its own.
    #[test]
    fn a_malformed_method_is_reported_once() {
        let out = expand(
            quote!(Flex, Fusion, Remote: cfg(feature = "remote")),
            quote! {
                pub trait Ext: Backend {
                    fn op(x: FloatTensor<Self>) -> FloatTensor<Self>;
                }
            },
        );
        assert_eq!(out.matches("compile_error").count(), 1, "{out}");
        assert!(out.contains("BackendRouter"), "{out}");
    }

    /// An argument the server cannot rebuild is reported once, by the client, which leaves the
    /// method out of the handlers. With `Remote` the only concrete backend, a method without
    /// tensor inputs still dispatches to it rather than adding an error about the backend count.
    #[test]
    fn a_remote_error_is_reported_once() {
        let load = quote! {
            #[fusion(default)]
            fn load(index: usize) -> FloatTensor<Self> { todo!() }
        };
        for (backends, load) in [
            (quote!(Flex, Fusion, Remote), quote!()),
            (quote!(Fusion, Remote), load),
        ] {
            let out = expand(
                backends.clone(),
                quote! {
                    pub trait Ext: Backend {
                        #[fusion(dtype = x, shape = x)]
                        fn op(x: FloatTensor<Self>, window: (usize, usize)) -> FloatTensor<Self>;
                        #load
                    }
                },
            );
            assert_eq!(out.matches("compile_error").count(), 1, "{backends}: {out}");
            assert!(out.contains("no scalar encoding"), "{backends}: {out}");
            assert!(
                !out.contains("registry . try_register"),
                "{backends}: {out}"
            );
        }
    }

    /// An ungated `Remote` keeps an ungated dispatch arm, rather than one behind an empty gate.
    #[test]
    fn ungated_remote_dispatch_stays_ungated() {
        let out = expand(
            quote!(Fusion, Remote),
            quote! {
                pub trait Ext: Backend {
                    #[fusion(dtype = x, shape = x)]
                    fn op(x: FloatTensor<Self>) -> FloatTensor<Self>;
                    #[fusion(default)]
                    fn load(index: usize) -> FloatTensor<Self> { todo!() }
                }
            },
        );
        assert!(!out.contains("match ()"), "{out}");
    }

    /// A scalar with a decoding callable is encoded through `encode_scalar` in the client only:
    /// the Fusion wrapper, which also builds without the remote backend, never names the callable.
    #[test]
    fn only_the_client_ties_an_encoding_to_its_decoding() {
        let out = expand(
            quote!(Flex, Fusion, Remote: cfg(feature = "remote")),
            quote! {
                pub trait Ext: Backend {
                    #[fusion(dtype = x, shape = x)]
                    fn op(
                        x: FloatTensor<Self>,
                        #[fusion(scalar = mode.to_code(), decode = remote_only::decode)] mode: Mode,
                    ) -> FloatTensor<Self>;
                }
            },
        );
        let client = out.find("BackendRouter < __R >").unwrap();
        let (fusion, client) = out.split_at(client);
        assert!(!fusion.contains("remote_only"), "{fusion}");
        assert!(client.contains("encode_scalar"), "{client}");
    }

    #[test]
    fn remote_without_fusion_stays_handwritten() {
        let out = expand(
            quote!(Remote),
            quote! {
                pub trait Ext: Backend {
                    fn op(x: FloatTensor<Self>, window: (usize, usize)) -> FloatTensor<Self>;
                }
            },
        );
        assert!(!out.contains("BackendRouter"));
        assert!(!out.contains("ExtHandlers"));
    }

    #[test]
    fn rejects_arguments_the_server_cannot_rebuild() {
        for (argument, message) in [
            (quote!(window: (usize, usize)), "no scalar encoding"),
            (quote!(scheme: QuantScheme), "no scalar encoding"),
            (
                quote!(#[fusion(scalar = mode.to_code())] mode: Mode),
                "add `decode = <callable>`",
            ),
        ] {
            let out = expand(
                quote!(Cube, Fusion, Remote),
                quote! {
                    pub trait Ext: Backend {
                        #[fusion(dtype = x, shape = x)]
                        fn op(x: FloatTensor<Self>, #argument) -> FloatTensor<Self>;
                    }
                },
            );
            assert!(out.contains("compile_error"), "accepted {argument}");
            assert!(out.contains(message), "{argument}: {out}");
        }
    }

    #[test]
    fn derives_remote_adapter_and_rejects_unencoded_fields() {
        let derive = |item| crate::derive::expand(item).unwrap().to_string();
        let out = derive(quote! {
            #[extension_type(fusion, remote)]
            pub enum Store<B: Backend> {
                Float { buffer: FloatTensor<B> },
                Packed {
                    indices: IntTensor<B>,
                    #[extension_type(scalar = bits.to_code(), decode = |code| Bits::from_code(code).unwrap())]
                    bits: Bits,
                    count: usize,
                    #[extension_type]
                    nested: Inner<B>,
                },
            }
        });
        assert!(out.contains("RemoteValueAdapter < B > for Store < B >"));
        assert!(out.contains("ScalarIr :: UInt (1u64)"));
        syn::parse_str::<syn::File>(&out).unwrap();

        let error = crate::derive::expand(quote! {
            #[extension_type(fusion, remote)]
            pub struct Store<B: Backend> { buffer: FloatTensor<B>, bits: Bits }
        })
        .unwrap()
        .to_string();
        assert!(error.contains("compile_error"));
        assert!(error.contains("no scalar"));

        assert!(
            crate::derive::expand(quote! {
                #[extension_type(remote)]
                pub struct Store<B: Backend> { buffer: FloatTensor<B> }
            })
            .is_err()
        );
    }
}
