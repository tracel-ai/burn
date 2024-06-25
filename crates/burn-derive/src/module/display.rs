use quote::quote;

pub fn attributes_fn(ast: &syn::DeriveInput) -> proc_macro2::TokenStream {
    match &ast.data {
        syn::Data::Struct(ref data_struct) => {
            let fields = match &data_struct.fields {
                syn::Fields::Named(ref named_fields) => {
                    named_fields.named.iter().collect::<Vec<_>>()
                }
                syn::Fields::Unit => Vec::new(),
                _ => panic!("attributes_fn only supports structs with named or unit fields"),
            };
            let field_prints = fields.iter().map(|field| {
                let field_name = &field.ident;
                quote! { .add(stringify!(#field_name), &self.#field_name) }
            });
            let struct_name = &ast.ident;
            quote! {
                fn content(&self, mut content: burn::module::Content) -> Option<burn::module::Content> {
                    content
                        .set_top_level_type(&stringify!(#struct_name))
                        #(#field_prints)*
                        .optional()
                }
            }
        }
        syn::Data::Enum(ref data_enum) => {
            let variant_prints = data_enum.variants.iter().map(|variant| {
                let variant_name = &variant.ident;
                match &variant.fields {
                    syn::Fields::Unit => {
                        quote! {
                            Self::#variant_name => {
                                content.add_formatted(&stringify!(#variant_name).to_string())
                                    .optional()

                            }
                        }
                    }
                    syn::Fields::Named(ref named_fields) => {
                        let field_prints = named_fields.named.iter().map(|field| {
                            let field_name = &field.ident;
                            quote! { .add(stringify!(#field_name), &self.#field_name) }
                        });

                        let field_names = named_fields.named.iter().map(|field| {
                            let field_name = &field.ident;
                            quote! { #field_name }
                        });

                        quote! {
                            Self::#variant_name { #(#field_names),* } => {
                                content.set_top_level_type(&stringify!(#variant_name))
                                #(#field_prints)*
                                .optional()
                            }
                        }
                    }
                    syn::Fields::Unnamed(ref unnamed_fields) => {
                        let field_names = (0..unnamed_fields.unnamed.len()).map(|i| {
                            syn::Ident::new(&format!("_{}", i), proc_macro2::Span::call_site())
                        });

                        let field_prints = field_names.clone().map(|field_name| {
                            quote! { .add(stringify!(#field_name), #field_name) }
                        });
                        quote! {
                            Self::#variant_name(#(#field_names),*) => {
                                content.set_top_level_type(&stringify!(#variant_name))
                                #(#field_prints)*
                                .optional()
                            }
                        }
                    }
                }
            });
            quote! {
                fn content(&self, mut content: burn::module::Content) -> Option<burn::module::Content> {
                    match self {
                        #(#variant_prints)*
                    }
                }
            }
        }
        _ => panic!("attributes_fn only supports structs and enums"),
    }
}

pub fn display_fn(_ast: &syn::DeriveInput) -> proc_macro2::TokenStream {
    quote! {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            let formatted = burn::module::ModuleDisplay::format(self, Default::default());
            write!(f, "{}", formatted)
        }
    }
}
