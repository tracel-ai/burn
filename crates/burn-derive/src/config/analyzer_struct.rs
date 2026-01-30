use super::ConfigAnalyzer;
use crate::shared::{attribute::AttributeItem, field::FieldTypeAnalyzer};
use proc_macro2::{Ident, TokenStream};
use quote::quote;

pub struct ConfigStructAnalyzer {
    name: Ident,
    fields_required: Vec<FieldTypeAnalyzer>,
    fields_option: Vec<FieldTypeAnalyzer>,
    fields_default: Vec<(FieldTypeAnalyzer, AttributeItem)>,
}

impl ConfigStructAnalyzer {
    pub fn new(
        name: Ident,
        fields_required: Vec<FieldTypeAnalyzer>,
        fields_option: Vec<FieldTypeAnalyzer>,
        fields_default: Vec<(FieldTypeAnalyzer, AttributeItem)>,
    ) -> Self {
        Self {
            name,
            fields_required,
            fields_option,
            fields_default,
        }
    }

    fn wrap_impl_block(&self, tokens: TokenStream) -> TokenStream {
        let name = &self.name;

        quote! {
            impl #name {
                #tokens
            }
        }
    }

    fn names(&self) -> Vec<FieldTypeAnalyzer> {
        let mut names = Vec::new();

        for field in self.fields_required.iter() {
            names.push(field.clone());
        }

        for field in self.fields_option.iter() {
            names.push(field.clone());
        }

        for (field, _) in self.fields_default.iter() {
            names.push(field.clone());
        }

        names
    }

    fn name_types(&self, names: &[FieldTypeAnalyzer]) -> Vec<TokenStream> {
        let mut name_types = Vec::new();

        for field in names.iter() {
            let name = field.ident();
            let ty = &field.field.ty;

            name_types.push(quote! {
                #name: #ty
            });
        }

        name_types
    }

    fn serde_struct_ident(&self) -> Ident {
        Ident::new(&format!("{}Serde", self.name), self.name.span())
    }

    fn gen_serialize_fn(
        &self,
        struct_name: &Ident,
        struct_gen: &TokenStream,
        names: &[FieldTypeAnalyzer],
    ) -> TokenStream {
        let name = &self.name;
        let names = names.iter().map(|name| {
            let name = name.ident();
            quote! { #name: self.#name.clone() }
        });

        quote! {
            impl burn::serde::Serialize for #name {

                fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
                where
                    S: burn::serde::Serializer {
                    #[derive(burn::serde::Serialize)]
                    #[serde(crate = "burn::serde")]
                    #struct_gen

                    let serde_state = #struct_name {
                        #(#names),*
                    };
                    serde_state.serialize(serializer)
                }
            }

        }
    }

    fn gen_deserialize_fn(
        &self,
        struct_name: &Ident,
        struct_gen: &TokenStream,
        names: &[FieldTypeAnalyzer],
    ) -> TokenStream {
        let name = &self.name;
        let names = names.iter().map(|name| {
            let name = name.ident();
            quote! { #name: serde_state.#name }
        });

        quote! {
            impl<'de> burn::serde::Deserialize<'de> for #name {
                fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
                where
                    D: burn::serde::Deserializer<'de> {
                    #[derive(burn::serde::Deserialize)]
                    #[serde(crate = "burn::serde")]
                    #struct_gen

                    let serde_state = #struct_name::deserialize(deserializer)?;
                    Ok(#name {
                        #(#names),*
                    })
                }
            }

        }
    }

    fn gen_serde_struct(&self, names: &[TokenStream]) -> TokenStream {
        let struct_name = self.serde_struct_ident();

        quote! {
            struct #struct_name {
                #(#names),*
            }

        }
    }
}

impl ConfigAnalyzer for ConfigStructAnalyzer {
    fn gen_new_fn(&self) -> TokenStream {
        let mut body = quote! {};
        let mut args = Vec::new();

        let mut fn_docs = quote! {};
        let mut has_field_docs = false;
        let mut has_required_docs = false;
        let mut has_option_docs = false;
        let mut has_default_docs = false;
        let mut docs_header = |fn_docs: &mut TokenStream,
                               required_docs: bool,
                               option_docs: bool,
                               default_docs: bool| {
            if !has_field_docs {
                has_field_docs = true;
                fn_docs.extend(quote! {
                    #[doc = "# Arguments"]
                });
            }
            if !has_required_docs && required_docs {
                fn_docs.extend(quote! {
                    #[doc = "###### Required Arguments"]
                });
                has_required_docs = true;
            }
            if !has_option_docs && option_docs {
                fn_docs.extend(quote! {
                    #[doc = "###### Optional Arguments"]
                });
                has_option_docs = true;
            }
            if !has_default_docs && default_docs {
                fn_docs.extend(quote! {
                    #[doc = "###### Default Arguments"]
                });
                has_default_docs = true;
            }
        };

        for field in self.fields_required.iter() {
            let name = field.ident();
            let ty = &field.field.ty;
            let docs = field.docs();

            body.extend(quote! {
                #name: #name,
            });
            args.push(quote! {
                #name: #ty
            });
            docs_header(&mut fn_docs, true, false, false);
            let doc_str = format!("###### `{}`\n\n", quote!(#name));
            fn_docs.extend(quote! {
                #[doc = #doc_str]
                #(#docs)*
            });
        }

        for field in self.fields_option.iter() {
            let name = field.ident();
            let docs = field.docs();

            body.extend(quote! {
                #name: None,
            });
            docs_header(&mut fn_docs, false, true, false);
            let default_doc = "- Defaults to `None`";
            let doc_str = format!("###### `{}`\n", quote!(#name));
            fn_docs.extend(quote! {
                #[doc = #doc_str]
                #(#docs)*
                #[doc = #default_doc]
            });
        }

        for (field, attribute) in self.fields_default.iter() {
            let name = field.ident();
            let value = &attribute.value;
            let docs = field.docs();

            match value {
                syn::Lit::Str(value) => {
                    let stream: proc_macro2::TokenStream = value.value().parse().unwrap();

                    body.extend(quote! {
                        #name: #stream,
                    });
                }
                _ => {
                    body.extend(quote! {
                        #name: #value,
                    });
                }
            };
            docs_header(&mut fn_docs, false, false, true);
            let default_doc = format!("- Defaults to `{}`", quote!(#value));
            let doc_str = format!("###### `{}`\n", quote!(#name));
            fn_docs.extend(quote! {
                #[doc = #doc_str]
                #(#docs)*
                #[doc = #default_doc]
            });
        }

        let body = quote! {
            #[doc = "Create a new instance of the config."]
            #fn_docs
            #[allow(clippy::too_many_arguments)]
            pub fn new(
                #(#args),*
            ) -> Self {
                Self { #body }
            }
        };
        self.wrap_impl_block(body)
    }

    fn gen_builder_fns(&self) -> TokenStream {
        let mut body = quote! {};

        for (field, attribute) in self.fields_default.iter() {
            let name = field.ident();
            let ty = &field.field.ty;
            let value = &attribute.value;
            let docs = field.docs();
            let doc_str = format!(
                "Sets the value for the field [`{}`](Self::{0}).\n- Defaults to `{}`\n\n",
                quote!(#name),
                quote!(#value)
            );
            let fn_docs = quote! {
                #[doc = #doc_str]
                #(#docs)*
            };
            let fn_name = Ident::new(&format!("with_{name}"), name.span());

            body.extend(quote! {
                #fn_docs
                pub fn #fn_name(mut self, #name: #ty) -> Self {
                    self.#name = #name;
                    self
                }
            });
        }

        for field in self.fields_option.iter() {
            let name = field.ident();
            let ty = &field.field.ty;
            let docs = field.docs();
            let doc_str = format!(
                "Sets the value for the field [`{}`](Self::{0}).\n- Defaults to `None`\n\n",
                quote!(#name)
            );
            let fn_docs = quote! {
                #[doc = #doc_str]
                #(#docs)*
            };
            let fn_name = Ident::new(&format!("with_{name}"), name.span());

            body.extend(quote! {
                #fn_docs
                pub fn #fn_name(mut self, #name: #ty) -> Self {
                    self.#name = #name;
                    self
                }
            });
        }

        self.wrap_impl_block(body)
    }

    fn gen_serde_impl(&self) -> TokenStream {
        let names = self.names();

        let struct_name = self.serde_struct_ident();
        let name_types = self.name_types(&names);
        let struct_gen = self.gen_serde_struct(&name_types);

        let serialize_gen = self.gen_serialize_fn(&struct_name, &struct_gen, &names);
        let deserialize_gen = self.gen_deserialize_fn(&struct_name, &struct_gen, &names);

        quote! {
            #serialize_gen
            #deserialize_gen
        }
    }

    fn gen_clone_impl(&self) -> TokenStream {
        let name = &self.name;
        let names = self.names().into_iter().map(|name| {
            let name = name.ident();
            quote! { #name: self.#name.clone() }
        });

        quote! {
            impl Clone for #name {
                fn clone(&self) -> Self {
                    Self {
                        #(#names),*
                    }
                }
            }

        }
    }

    fn gen_display_impl(&self) -> TokenStream {
        let name = &self.name;

        quote! {
            impl core::fmt::Display for #name {
                fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                    f.write_str(&burn::config::config_to_json(self))
                }
            }
        }
    }

    fn gen_config_impl(&self) -> TokenStream {
        let name = &self.name;

        quote! {
            impl burn::config::Config for #name {
            }
        }
    }
}
