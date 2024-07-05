use proc_macro::TokenStream;
use quote::quote;
use syn::Ident;

use super::GenericsCodegen;

struct TypeCodegen {
    name: syn::Ident,
    name_launch: syn::Ident,
    name_expand: syn::Ident,
    fields: Vec<syn::Field>,
    generics: GenericsCodegen,
    vis: syn::Visibility,
}

impl TypeCodegen {
    pub fn expand_ty(&self) -> proc_macro2::TokenStream {
        let mut fields = quote::quote! {};
        let name = &self.name_expand;

        for field in self.fields.iter() {
            let ident = &field.ident;
            let ty = &field.ty;
            let vis = &field.vis;

            fields.extend(quote! {
                #vis #ident: <#ty as CubeType>::ExpandType,
            });
        }

        let generics = self.generics.type_definitions();
        let vis = &self.vis;

        quote! {
            #[derive(Clone)]
            #vis struct #name #generics {
                #fields
            }
        }
    }

    pub fn launch_ty(&self) -> proc_macro2::TokenStream {
        let mut fields = quote::quote! {};
        let name = &self.name_launch;

        for field in self.fields.iter() {
            let ident = &field.ident;
            let ty = &field.ty;
            let vis = &field.vis;

            fields.extend(quote! {
                #vis #ident: <#ty as LaunchArg>::RuntimeArg<'a, R>,
            });
        }

        let generics = self.generics.all_definitions();

        quote! {
            struct #name #generics {
                #fields
            }
        }
    }

    pub fn launch_new(&self) -> proc_macro2::TokenStream {
        let mut args = quote::quote! {};
        let mut fields = quote::quote! {};
        let name = &self.name_launch;

        for field in self.fields.iter() {
            let ident = &field.ident;
            let ty = &field.ty;
            let vis = &field.vis;

            args.extend(quote! {
                #vis #ident: <#ty as LaunchArg>::RuntimeArg<'a, R>,
            });
            fields.extend(quote! {
                #ident,
            });
        }

        let generics_impl = self.generics.all_definitions();
        let generics_use = self.generics.all_in_use();
        let vis = &self.vis;

        quote! {
            impl #generics_impl #name #generics_use {
                /// New kernel
                #[allow(clippy::too_many_arguments)]
                #vis fn new(#args) -> Self {
                    Self {
                        #fields
                    }
                }
            }
        }
    }

    pub fn arg_settings_impl(&self) -> proc_macro2::TokenStream {
        let mut register_body = quote::quote! {};
        let mut configure_input_body = quote::quote! {};
        let mut configure_output_body = quote::quote! {};
        let name = &self.name_launch;

        for (pos, field) in self.fields.iter().enumerate() {
            let ident = &field.ident;

            register_body.extend(quote! {
                self.#ident.register(launcher);
            });
            configure_input_body.extend(quote! {
                settings = ArgSettings::<R>::configure_input(&self.#ident, #pos, settings);
            });
            configure_output_body.extend(quote! {
                settings = ArgSettings::<R>::configure_output(&self.#ident, #pos, settings);
            });
        }

        let generics_impl = self.generics.all_definitions();
        let generics_use = self.generics.all_in_use();

        quote! {
            impl #generics_impl ArgSettings<R> for #name #generics_use {
                fn register(&self, launcher: &mut KernelLauncher<R>) {
                    #register_body
                }

                fn configure_input(&self, position: usize, mut settings: KernelSettings) -> KernelSettings {
                    #configure_input_body

                    settings
                }

                fn configure_output(&self, position: usize, mut settings: KernelSettings) -> KernelSettings {
                    #configure_output_body

                    settings
                }
            }
        }
    }

    pub fn cube_type_impl(&self) -> proc_macro2::TokenStream {
        let name = &self.name;
        let name_expand = &self.name_expand;

        let generics_impl = self.generics.type_definitions();
        let generics_use = self.generics.type_in_use();

        quote! {
            impl #generics_impl CubeType for #name #generics_use {
                type ExpandType = #name_expand #generics_use;
            }
        }
    }

    pub fn launch_arg_impl(&self) -> proc_macro2::TokenStream {
        let mut body_input = quote::quote! {};
        let mut body_output = quote::quote! {};
        let name = &self.name;
        let name_launch = &self.name_launch;
        let name_expand = &self.name_expand;

        for field in self.fields.iter() {
            let ident = &field.ident;
            let ty = &field.ty;
            let vis = &field.vis;

            body_input.extend(quote! {
                #vis #ident: <#ty as LaunchArgExpand>::expand(builder, vectorization),
            });
            body_output.extend(quote! {
                #vis #ident: <#ty as LaunchArgExpand>::expand_output(builder, vectorization),
            });
        }

        let type_generics_impl = self.generics.type_definitions();
        let type_generics_use = self.generics.type_in_use();

        let runtime_generics_impl = self.generics.runtime_definitions();
        let all_generics_use = self.generics.all_in_use();

        quote! {
            impl #type_generics_impl LaunchArg for #name #type_generics_use {
                type RuntimeArg #runtime_generics_impl = #name_launch #all_generics_use;
            }

            impl #type_generics_impl LaunchArgExpand for #name #type_generics_use {
                fn expand(
                    builder: &mut KernelBuilder,
                    vectorization: burn_cube::ir::Vectorization,
                ) -> <Self as CubeType>::ExpandType {
                    #name_expand {
                        #body_input
                    }
                }
                fn expand_output(
                    builder: &mut KernelBuilder,
                    vectorization: burn_cube::ir::Vectorization,
                ) -> <Self as CubeType>::ExpandType {
                    #name_expand {
                        #body_output
                    }
                }
            }
        }
    }

    pub fn expand_type_impl(&self) -> proc_macro2::TokenStream {
        let name_expand = &self.name_expand;
        let type_generics_impl = self.generics.type_definitions();
        let type_generics_use = self.generics.type_in_use();

        let mut body = quote::quote! {};
        for field in self.fields.iter() {
            let ident = &field.ident;
            body.extend(quote::quote! {
                #ident: Init::init(self.#ident, context),
            });
        }

        quote! {
            impl #type_generics_impl Init for #name_expand  #type_generics_use {
                fn init(self, context: &mut CubeContext) -> Self {
                    Self {
                        #body
                    }
                }
            }
        }
    }
}

pub(crate) fn generate_cube_type(ast: &syn::DeriveInput, with_launch: bool) -> TokenStream {
    let name = ast.ident.clone();
    let generics = ast.generics.clone();
    let visibility = ast.vis.clone();

    let name_string = name.to_string();
    let name_expand = Ident::new(format!("{}Expand", name_string).as_str(), name.span());
    let name_launch = Ident::new(format!("{}Launch", name_string).as_str(), name.span());

    let mut fields = Vec::new();

    match &ast.data {
        syn::Data::Struct(struct_data) => {
            for field in struct_data.fields.iter() {
                fields.push(field.clone());
            }
        }
        syn::Data::Enum(_) => panic!("Only struct can be derived"),
        syn::Data::Union(_) => panic!("Only struct can be derived"),
    };

    let codegen = TypeCodegen {
        name,
        name_launch,
        name_expand,
        fields,
        generics: GenericsCodegen::new(generics),
        vis: visibility,
    };

    let expand_ty = codegen.expand_ty();
    let launch_ty = codegen.launch_ty();
    let launch_new = codegen.launch_new();

    let cube_type_impl = codegen.cube_type_impl();
    let arg_settings_impl = codegen.arg_settings_impl();
    let launch_arg_impl = codegen.launch_arg_impl();
    let expand_type_impl = codegen.expand_type_impl();

    if with_launch {
        quote! {
            #expand_ty
            #launch_ty
            #launch_new

            #cube_type_impl
            #arg_settings_impl
            #launch_arg_impl
            #expand_type_impl
        }
        .into()
    } else {
        quote! {
            #expand_ty
            #cube_type_impl
            #expand_type_impl
        }
        .into()
    }
}
