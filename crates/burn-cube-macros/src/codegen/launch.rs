use proc_macro2::{Span, TokenStream};
use syn::{parse_quote, Generics, Ident};

#[derive(Default)]
struct KernelStructCodegen {
    name: String,
    generics: Generics,
    inputs: Vec<(Ident, syn::Type)>,
    outputs: Vec<(Ident, syn::Type)>,
    comptimes: Vec<(syn::Type, Ident)>,
    args: Vec<TokenStream>,
}

impl KernelStructCodegen {
    pub fn gen_kernel(&self) -> TokenStream {
        let ident = Ident::new(&self.name, Span::call_site());
        let generics = add_runtime(self.generics.clone());
        let phantoms = self.phantoms(&generics, true);
        let mut comptimes = quote::quote! {};

        for (ty, ident) in self.comptimes.iter() {
            comptimes.extend(quote::quote! {
                #ident: #ty,
            });
        }

        quote::quote! {
            pub struct #ident #generics {
                settings: CompilationSettings,
                #comptimes
                #phantoms
            }
        }
    }
    pub fn gen_compile(&self, expand: &Ident) -> TokenStream {
        let ident = Ident::new(&self.name, Span::call_site());
        let generics = add_runtime(self.generics.clone());
        let (impl_gen, ty_gen, where_gen) = generics.split_for_impl();

        let mut variables = quote::quote! {};

        for (ident, ty) in self.inputs.iter() {
            variables.extend(quote::quote! {
                let #ident = <#ty as LaunchArg>::compile_input(&mut builder);
            });
        }

        for (ident, ty) in self.outputs.iter() {
            variables.extend(quote::quote! {
                let #ident = <#ty as LaunchArg>::compile_output(&mut builder);
            });
        }

        let mut expand_args = quote::quote! { &mut builder.context, };

        for arg in self.args.iter() {
            expand_args.extend(quote::quote! {
                #arg,
            })
        }

        let generics = self.generics.split_for_impl().1;

        let mut format_str = "{:?}-{}".to_string();
        for _ in 0..self.comptimes.len() {
            format_str.push_str("-{:?}");
        }

        let mut format_args = quote::quote! { core::any::TypeId::of::<Self>(), self.settings, };

        for (_, ident) in self.comptimes.iter() {
            format_args.extend(quote::quote! { self.#ident, });
        }

        quote::quote! {
            impl #impl_gen GpuComputeShaderPhase for #ident #ty_gen #where_gen {
                fn compile(&self) -> ComputeShader {
                    let mut builder = KernelBuilder::default();

                    #variables

                    #expand::#generics(#expand_args);

                    let compilation = builder.build();
                    compilation.compile(self.settings.clone())
                }

                fn id(&self) -> String {
                    format!(#format_str, #format_args)
                }
            }
        }
    }

    pub fn phantoms(&self, generics: &Generics, declaration: bool) -> TokenStream {
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

    pub fn gen_execution(&self) -> TokenStream {
        let ident = Ident::new(&self.name, Span::call_site());
        let generics = add_runtime(self.generics.clone());
        let phantoms = self.phantoms(&generics, false);

        let mut comptimes = quote::quote! {};
        let mut body = quote::quote! {
            let mut launcher = KernelLauncher::<R>::default();
        };

        for (input, _) in self.inputs.iter() {
            body.extend(quote::quote! {
                #input.register(&mut launcher);
            });
        }

        for (input, _) in self.outputs.iter() {
            body.extend(quote::quote! {
                #input.register(&mut launcher);
            });
        }

        for (_ty, ident) in self.comptimes.iter() {
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

            launcher.launch(workgroup, kernel, client);
        }
        .into()
    }
}

pub fn codegen_launch(sig: &syn::Signature) -> TokenStream {
    let mut struct_codegen = KernelStructCodegen::default();
    let mut first_letter = sig.ident.to_string();
    let second_part = first_letter.split_off(1);

    struct_codegen.name = format!("{}{}", first_letter.to_uppercase(), second_part);
    struct_codegen.generics = sig.generics.clone();

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

                        match pat.ty.as_ref() {
                            syn::Type::Path(pat) => {
                                if let Some(name) = pat.path.segments.first() {
                                    let name = name.ident.to_string();

                                    if name == "Comptime" {
                                        comptime = true;
                                    }
                                }
                            }
                            _ => (),
                        };

                        (pat.ty.clone(), ident.ident.clone())
                    }
                    _ => panic!("Nop"),
                };

                if comptime {
                    struct_codegen.args.push(quote::quote! {
                        self.#ident
                    });
                } else {
                    struct_codegen.args.push(quote::quote! {
                        #ident
                    });
                }

                if comptime {
                    inputs.extend(quote::quote! {
                        #ident: <#ty as burn_cube::CubeType>::ExpandType,
                    });
                } else {
                    inputs.extend(quote::quote! {
                        #ident: RuntimeArg<'a, #ty, R>,
                    });
                }

                if is_output {
                    struct_codegen.outputs.push((ident.clone(), *ty));
                } else if comptime {
                    let ty = first_generic_ty(&ty);
                    struct_codegen.comptimes.push((ty.clone(), ident.clone()));
                } else {
                    struct_codegen.inputs.push((ident.clone(), *ty));
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
                <#ty as burn_cube::CubeType>::ExpandType
            });
        }
    }

    let ident = &sig.ident;
    let ident_expand = syn::Ident::new(format!("{ident}_expand").as_str(), ident.span());
    let ident = syn::Ident::new(format!("{ident}_launch").as_str(), ident.span());

    let generics = add_runtime(add_lifetime(sig.generics.clone()));
    let body = struct_codegen.gen_execution();
    let kernel = struct_codegen.gen_kernel();
    let compile = struct_codegen.gen_compile(&ident_expand);

    quote::quote! {
        #kernel
        #compile

        pub fn #ident #generics (
            client: ComputeClient<R::Server, R::Channel>,
            workgroup: WorkGroup,
            settings: CompilationSettings,
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
