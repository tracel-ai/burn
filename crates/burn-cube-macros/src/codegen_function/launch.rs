use proc_macro2::{Span, TokenStream};
use syn::{parse_quote, Generics, Ident};

#[derive(Default)]
struct Codegen {
    // Basic attributes.
    name: String,
    generics: Generics,
    fn_inputs: TokenStream,
    fn_output: TokenStream,
    // States to generate code.
    state_comptimes: Vec<(syn::Type, Ident)>,
    state_args: Vec<TokenStream>,
    state_inputs: Vec<(Ident, syn::Type)>,
    state_outputs: Vec<(Ident, syn::Type)>,
}

impl Codegen {
    fn from_sig(sig: &syn::Signature) -> Self {
        let mut codegen = Codegen::default();

        let mut first_letter = sig.ident.to_string();
        let second_part = first_letter.split_off(1);

        codegen.name = format!("{}{}", first_letter.to_uppercase(), second_part);
        codegen.generics = sig.generics.clone();

        let mut inputs = quote::quote!();

        for input in &sig.inputs {
            let mut is_output = false;
            let mut comptime = false;

            match input {
                syn::FnArg::Typed(pat) => {
                    let (ty, ident) = match pat.pat.as_ref() {
                        syn::Pat::Ident(ident) => {
                            if ident.mutability.is_some() {
                                is_output = true;
                            }

                            if let syn::Type::Path(pat) = pat.ty.as_ref() {
                                if let Some(name) = pat.path.segments.first() {
                                    let name = name.ident.to_string();

                                    if name == "Comptime" {
                                        comptime = true;
                                    }
                                }
                            };

                            (pat.ty.clone(), ident.ident.clone())
                        }
                        _ => panic!("Nop"),
                    };

                    if comptime {
                        codegen.state_args.push(quote::quote! {
                            self.#ident
                        });
                    } else {
                        codegen.state_args.push(quote::quote! {
                            #ident
                        });
                    }

                    if comptime {
                        inputs.extend(quote::quote! {
                            #ident: <#ty as burn_cube::frontend::CubeType>::ExpandType,
                        });
                    } else {
                        inputs.extend(quote::quote! {
                            #ident: RuntimeArg<'a, #ty, R>,
                        });
                    }

                    if is_output {
                        codegen.state_outputs.push((ident.clone(), *ty));
                    } else if comptime {
                        let ty = first_generic_ty(&ty);
                        codegen.state_comptimes.push((ty.clone(), ident.clone()));
                    } else {
                        codegen.state_inputs.push((ident.clone(), *ty));
                    }
                }
                _ => todo!("Only Typed inputs are supported"),
            };
        }

        let mut output = quote::quote!();

        match &sig.output {
            syn::ReturnType::Default => output.extend(quote::quote! {()}),
            syn::ReturnType::Type(_, ty) => {
                output.extend(quote::quote! {
                    <#ty as burn_cube::frontend::CubeType>::ExpandType
                });
            }
        }

        codegen.fn_inputs = inputs;
        codegen.fn_output = output;

        codegen
    }

    fn gen_kernel_struct(&self) -> TokenStream {
        let ident = Ident::new(&self.name, Span::call_site());
        let generics = add_runtime(self.generics.clone());
        let phantoms = self.phantoms(&generics, true);
        let mut comptimes = quote::quote! {};

        for (ty, ident) in self.state_comptimes.iter() {
            comptimes.extend(quote::quote! {
                #ident: #ty,
            });
        }

        quote::quote! {
            /// Kernel
            pub struct #ident #generics {
                settings: KernelSettings,
                #comptimes
                #phantoms
            }
        }
    }

    fn gen_compile_impl(&self, expand: &Ident) -> TokenStream {
        let ident = Ident::new(&self.name, Span::call_site());
        let generics = add_runtime(self.generics.clone());
        let (impl_gen, ty_gen, where_gen) = generics.split_for_impl();

        let mut variables = quote::quote! {};

        for (pos, (ident, ty)) in self.state_inputs.iter().enumerate() {
            variables.extend(quote::quote! {
                let #ident = <#ty as LaunchArg>::compile_input(&mut builder, self.settings.vectorization_input(#pos));
            });
        }

        for (pos, (ident, ty)) in self.state_outputs.iter().enumerate() {
            variables.extend(quote::quote! {
                let #ident = <#ty as LaunchArg>::compile_output(&mut builder, self.settings.vectorization_output(#pos));
            });
        }

        let mut expand_args = quote::quote! { &mut builder.context, };

        for arg in self.state_args.iter() {
            expand_args.extend(quote::quote! {
                #arg,
            })
        }

        let generics = self.generics.split_for_impl().1;

        let mut format_str = "{:?}-{}".to_string();
        for _ in 0..self.state_comptimes.len() {
            format_str.push_str("-{:?}");
        }

        let mut format_args = quote::quote! { core::any::TypeId::of::<Self>(), self.settings, };

        for (_, ident) in self.state_comptimes.iter() {
            format_args.extend(quote::quote! { self.#ident, });
        }

        quote::quote! {
            impl #impl_gen Kernel for #ident #ty_gen #where_gen {
                fn define(&self) -> KernelDefinition {
                    let mut builder = KernelBuilder::default();

                    #variables

                    #expand::#generics(#expand_args);

                    builder.build(self.settings.clone())
                }

                fn id(&self) -> String {
                    format!(#format_str, #format_args)
                }
            }
        }
    }

    fn phantoms(&self, generics: &Generics, declaration: bool) -> TokenStream {
        let mut phantoms = quote::quote! {};

        for param in generics.params.iter() {
            let ty = match param {
                syn::GenericParam::Type(ty) => ty,
                _ => continue,
            };
            let ident = Ident::new(
                format!("_{}", ty.ident.to_string().to_lowercase()).as_str(),
                Span::call_site(),
            );
            let ty = &ty.ident;
            if declaration {
                phantoms.extend(quote::quote! {
                    #ident: core::marker::PhantomData<#ty>,
                });
            } else {
                phantoms.extend(quote::quote! {
                    #ident: core::marker::PhantomData::<#ty>,
                });
            }
        }
        phantoms
    }

    fn gen_launch_body(&self) -> TokenStream {
        let ident = Ident::new(&self.name, Span::call_site());
        let generics = add_runtime(self.generics.clone());
        let phantoms = self.phantoms(&generics, false);

        let mut comptimes = quote::quote! {};
        let mut body = quote::quote! {
            let mut launcher = KernelLauncher::<R>::default();
        };

        for (input, _) in self.state_inputs.iter() {
            body.extend(quote::quote! {
                #input.register(&mut launcher);
            });
        }

        for (input, _) in self.state_outputs.iter() {
            body.extend(quote::quote! {
                #input.register(&mut launcher);
            });
        }

        for (_ty, ident) in self.state_comptimes.iter() {
            comptimes.extend(quote::quote! {
                #ident,
            });
        }

        let kernel = quote::quote! {
            #ident {
                settings,
                #comptimes
                #phantoms
            }
        };

        quote::quote! {
            let kernel = #kernel;

            #body

            launcher.launch(cube_count, kernel, client);
        }
    }
}

pub fn codegen_launch(sig: &syn::Signature) -> TokenStream {
    let codegen = Codegen::from_sig(sig);

    let ident = &sig.ident;
    let ident_expand = syn::Ident::new(format!("{ident}_expand").as_str(), ident.span());
    let ident = syn::Ident::new(format!("{ident}_launch").as_str(), ident.span());

    let generics = add_runtime(add_lifetime(sig.generics.clone()));
    let body = codegen.gen_launch_body();
    let kernel = codegen.gen_kernel_struct();
    let compile = codegen.gen_compile_impl(&ident_expand);
    let (inputs, output) = (codegen.fn_inputs, codegen.fn_output);

    quote::quote! {
        #kernel
        #compile

        #[allow(clippy::too_many_arguments)]
        /// Launch
        pub fn #ident #generics (
            client: ComputeClient<R::Server, R::Channel>,
            cube_count: CubeCount,
            settings: KernelSettings,
            #inputs
        ) -> #output {
            #body;
        }
    }
}

pub fn add_lifetime(mut generics: Generics) -> Generics {
    let lifetime: syn::Generics = parse_quote! {<'a>};

    generics
        .params
        .insert(0, lifetime.params.into_iter().next().unwrap());
    generics
}

pub fn add_runtime(mut generics: Generics) -> Generics {
    let runtime: syn::Generics = parse_quote! { <R: Runtime> };

    generics
        .params
        .push(runtime.params.into_iter().next().unwrap());
    generics
}

fn first_generic_ty(ty: &syn::Type) -> syn::Type {
    match ty {
        syn::Type::Path(pat) => match &pat.path.segments.first().unwrap().arguments {
            syn::PathArguments::AngleBracketed(ty) => match ty.args.first().unwrap() {
                syn::GenericArgument::Type(ty) => ty.clone(),
                _ => panic!("Should have a generic type"),
            },
            _ => panic!("Comptime must have a generic"),
        },
        _ => todo!(),
    }
}
