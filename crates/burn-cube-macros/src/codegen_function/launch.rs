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

                            if let syn::Type::Reference(ty) = pat.ty.as_ref() {
                                if ty.mutability.is_some() {
                                    is_output = true;
                                }
                            };

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
                        let ty = no_ref(&ty);
                        inputs.extend(quote::quote! {
                            #ident: <#ty as burn_cube::frontend::CubeType>::ExpandType,
                        });
                    } else {
                        let ty = no_ref(&ty);
                        inputs.extend(quote::quote! {
                            #ident: RuntimeArg<'a, #ty, R>,
                        });
                    }

                    if is_output {
                        codegen
                            .state_outputs
                            .push((ident.clone(), no_ref(&ty).clone()));
                    } else if comptime {
                        codegen
                            .state_comptimes
                            .push((first_generic_ty(&ty).clone(), ident.clone()));
                    } else {
                        codegen
                            .state_inputs
                            .push((ident.clone(), no_ref(&ty).clone()));
                    }
                }
                _ => panic!("Only Typed inputs are supported"),
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

    fn gen_settings(&self) -> TokenStream {
        let mut variables = quote::quote! {};

        for (pos, (ident, _ty)) in self.state_inputs.iter().enumerate() {
            variables.extend(quote::quote! {
                settings = ArgSettings::<R>::configure_input(&#ident, #pos, settings);
            });
        }

        for (pos, (ident, _ty)) in self.state_outputs.iter().enumerate() {
            variables.extend(quote::quote! {
                settings = ArgSettings::<R>::configure_output(&#ident, #pos, settings);
            });
        }

        quote::quote! {
            let mut settings = KernelSettings::default();
            settings = settings.cube_dim(cube_dim);
            #variables
        }
    }

    fn gen_register_input(&self) -> TokenStream {
        let generics = &self.generics;
        let mut variables = quote::quote! {};

        for (pos, (_ident, ty)) in self.state_inputs.iter().enumerate() {
            variables.extend(quote::quote! {
                #pos => std::sync::Arc::new(<#ty as LaunchArgExpand>::expand(builder, settings.vectorization_input(#pos))),
            });
        }

        quote::quote! {
            #[allow(unused)]
            fn register_input #generics(
                builder: &mut KernelBuilder,
                settings: &KernelSettings,
                position: usize,
            ) -> std::sync::Arc<dyn core::any::Any> {
                match position {
                    #variables
                    _ => panic!("Input {position} is invalid."),
                }
            }
        }
    }

    fn gen_register_output(&self) -> TokenStream {
        let generics = &self.generics;
        let mut variables = quote::quote! {};

        for (pos, (_ident, ty)) in self.state_outputs.iter().enumerate() {
            variables.extend(quote::quote! {
                #pos => std::sync::Arc::new(<#ty as LaunchArgExpand>::expand_output(builder, settings.vectorization_output(#pos))),
            });
        }

        quote::quote! {
            #[allow(unused)]
            fn register_output #generics (
                builder: &mut KernelBuilder,
                settings: &KernelSettings,
                position: usize,
            ) -> std::sync::Arc<dyn core::any::Any> {
                match position {
                    #variables
                    _ => panic!("Input {position} is invalid."),
                }
            }
        }
    }

    fn gen_define_impl(&self, expand: &TokenStream) -> TokenStream {
        let mut expand_args = quote::quote! { &mut builder.context, };

        let mut variables = quote::quote! {};

        for (pos, (ident, ty)) in self.state_inputs.iter().enumerate() {
            variables.extend(quote::quote! {
                let #ident: &<#ty as CubeType>::ExpandType = inputs
                    .get(&#pos)
                    .unwrap()
                    .downcast_ref()
                    .expect("Input type should be correct. It could be caused by an invalid kernel input/output alias.");
            });
        }

        for (pos, (ident, ty)) in self.state_outputs.iter().enumerate() {
            variables.extend(quote::quote! {
                let #ident: &<#ty as CubeType>::ExpandType = outputs
                    .get(&#pos)
                    .unwrap()
                    .downcast_ref()
                    .expect("Output type should be correct. It could be caused by an invalid kernel input/output alias.");
            });
        }

        for arg in self.state_args.iter() {
            expand_args.extend(quote::quote! {
                #arg.clone(),
            })
        }

        let expand_func = match self.generics.params.is_empty() {
            true => quote::quote! { #expand },
            false => {
                let generics = self.generics.split_for_impl().1;
                quote::quote! { #expand::#generics }
            }
        };

        quote::quote! {
            #variables
            #expand_func(#expand_args);
            builder.build(self.settings.clone())
        }
    }

    fn gen_define_args(&self) -> TokenStream {
        let num_inputs = self.state_inputs.len();
        let num_outputs = self.state_outputs.len();

        let register_input = self.gen_register_input();
        let register_output = self.gen_register_output();

        let (register_input_call, register_output_call) = match self.generics.params.is_empty() {
            true => (
                quote::quote! { register_input },
                quote::quote! { register_output },
            ),
            false => {
                let generics = self.generics.split_for_impl().1;

                (
                    quote::quote! { register_input::#generics },
                    quote::quote! { register_output::#generics },
                )
            }
        };

        let mut variables = quote::quote! {};

        for (pos, (ident, ty)) in self.state_inputs.iter().enumerate() {
            variables.extend(quote::quote! {
                let #ident = <&#ty as CubeType>::ExpandType =
                    *inputs.remove(&#pos).unwrap().downcast().unwrap();
            });
        }

        for (pos, (ident, ty)) in self.state_outputs.iter().enumerate() {
            variables.extend(quote::quote! {
                let #ident = <&mut #ty as CubeType>::ExpandType =
                    *outputs.remove(&#pos).unwrap().downcast().unwrap();
            });
        }

        let mut tokens = quote::quote! {
            let mut builder = KernelBuilder::default();

            let mut inputs: std::collections::BTreeMap<usize, std::sync::Arc<dyn core::any::Any>> = std::collections::BTreeMap::new();
            let mut outputs: std::collections::BTreeMap<usize, std::sync::Arc<dyn core::any::Any>> = std::collections::BTreeMap::new();

            for mapping in self.settings.mappings.iter() {
                if !inputs.contains_key(&mapping.pos_input) {
                    inputs.insert(
                        mapping.pos_input,
                        #register_input_call(&mut builder, &self.settings, mapping.pos_input),
                    );
                }

                let input = inputs.get(&mapping.pos_input).unwrap();
                outputs.insert(mapping.pos_output, input.clone());
            }

            #register_input
            #register_output
        };

        if num_inputs > 0 {
            tokens.extend(quote::quote! {
                for i in 0..#num_inputs {
                    if !inputs.contains_key(&i) {
                        inputs.insert(i, #register_input_call(&mut builder, &self.settings, i));
                    }
                }
            });
        }

        if num_outputs > 0 {
            tokens.extend(quote::quote! {
                for i in 0..#num_outputs {
                    if !outputs.contains_key(&i) {
                        outputs.insert(i, #register_output_call(&mut builder, &self.settings, i));
                    }
                }
            });
        }

        tokens
    }

    fn gen_compile_impl(&self, expand: &TokenStream) -> TokenStream {
        let ident = Ident::new(&self.name, Span::call_site());
        let generics = add_runtime(self.generics.clone());
        let (impl_gen, ty_gen, where_gen) = generics.split_for_impl();

        let mut format_str = "{:?}-{}".to_string();
        for _ in 0..self.state_comptimes.len() {
            format_str.push_str("-{:?}");
        }

        let mut format_args = quote::quote! { core::any::TypeId::of::<Self>(), self.settings, };

        for (_, ident) in self.state_comptimes.iter() {
            format_args.extend(quote::quote! { self.#ident, });
        }

        let define_args = self.gen_define_args();
        let define_impl = self.gen_define_impl(expand);

        quote::quote! {
            impl #impl_gen Kernel for #ident #ty_gen #where_gen {
                fn define(&self) -> KernelDefinition {
                    #define_args
                    #define_impl
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
        let settings = self.gen_settings();

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
            #settings

            let kernel = #kernel;

            #body

            launcher.launch(cube_count, kernel, client);
        }
    }
}

pub fn codegen_launch(sig: &syn::Signature) -> TokenStream {
    let codegen = Codegen::from_sig(sig);

    let ident = &sig.ident;

    let ident_expand = quote::quote! {
        __expand
    };

    let generics = add_runtime(add_lifetime(sig.generics.clone()));
    let body = codegen.gen_launch_body();
    let kernel = codegen.gen_kernel_struct();
    let compile = codegen.gen_compile_impl(&ident_expand);
    let (inputs, output) = (codegen.fn_inputs, codegen.fn_output);
    let doc = format!("Launch the kernel [{ident}()] on the given runtime.");

    quote::quote! {
        #kernel
        #compile

        #[allow(clippy::too_many_arguments)]
        #[doc = #doc]
        pub fn launch #generics (
            client: ComputeClient<R::Server, R::Channel>,
            cube_count: CubeCount<R::Server>,
            cube_dim: CubeDim,
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

fn no_ref(ty: &syn::Type) -> &syn::Type {
    match ty {
        syn::Type::Reference(val) => &val.elem,
        _ => ty,
    }
}
