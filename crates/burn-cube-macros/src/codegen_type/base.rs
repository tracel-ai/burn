use proc_macro::TokenStream;
use quote::quote;
use syn::Ident;

struct TypeCodegen {
    name: syn::Ident,
    name_launch: syn::Ident,
    name_expand: syn::Ident,
    fields: Vec<syn::Field>,
}

impl TypeCodegen {
    pub fn expand_ty(&self) -> proc_macro2::TokenStream {
        let mut fields = quote::quote! {};
        let name = &self.name_expand;

        for field in self.fields.iter() {
            let ident = &field.ident;
            let ty = &field.ty;

            fields.extend(quote! {
                #ident: <#ty as CubeType>::ExpandType,
            });
        }

        quote! {
            #[derive(Clone)]
            struct #name {
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

            fields.extend(quote! {
                #ident: <#ty as LaunchArg>::RuntimeArg<'a, R>,
            });
        }

        quote! {
            struct #name<'a, R: Runtime> {
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

            args.extend(quote! {
                #ident: <#ty as LaunchArg>::RuntimeArg<'a, R>,
            });
            fields.extend(quote! {
                #ident,
            });
        }

        quote! {
            impl<'a, R: Runtime> #name<'a, R> {
                pub fn new(#args) -> Self {
                    Self {
                        #fields
                    }
                }
            }
        }
    }

    pub fn arg_settings_impl(&self) -> proc_macro2::TokenStream {
        let mut body = quote::quote! {};
        let name = &self.name_launch;

        for field in self.fields.iter() {
            let ident = &field.ident;

            body.extend(quote! {
                self.#ident.register(launcher);
            });
        }

        quote! {
            impl<'a, R: Runtime> ArgSettings<R> for #name<'a, R> {
                fn register(&self, launcher: &mut KernelLauncher<R>) {
                    #body
                }
            }
        }
    }

    pub fn cube_type_impl(&self) -> proc_macro2::TokenStream {
        let name = &self.name;
        let name_expand = &self.name_expand;

        quote! {
            impl CubeType for #name {
                type ExpandType = #name_expand;
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

            body_input.extend(quote! {
                #ident: <#ty as LaunchArg>::compile_input(builder, vectorization),
            });
            body_output.extend(quote! {
                #ident: <#ty as LaunchArg>::compile_output(builder, vectorization),
            });
        }

        quote! {
            impl LaunchArg for #name {
                type RuntimeArg<'a, R: Runtime> = #name_launch<'a, R>;

                fn compile_input(
                    builder: &mut KernelBuilder,
                    vectorization: burn_cube::ir::Vectorization,
                ) -> <Self as CubeType>::ExpandType {
                    #name_expand {
                        #body_input
                    }
                }

                fn compile_output(
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
}

pub(crate) fn generate_cube_type(ast: &syn::DeriveInput) -> TokenStream {
    let name = ast.ident.clone();
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
    };

    let expand_ty = codegen.expand_ty();
    let launch_ty = codegen.launch_ty();
    let launch_new = codegen.launch_new();

    let cube_type_impl = codegen.cube_type_impl();
    let arg_settings_impl = codegen.arg_settings_impl();
    let launch_arg_impl = codegen.launch_arg_impl();

    quote! {
        #expand_ty
        #launch_ty
        #launch_new

        #cube_type_impl
        #arg_settings_impl
        #launch_arg_impl
    }
    .into()
}
