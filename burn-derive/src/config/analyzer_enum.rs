use super::ConfigAnalyzer;
use proc_macro2::{Ident, TokenStream};
use quote::quote;
use syn::{FieldsNamed, Variant};

pub struct ConfigEnumAnalyzer {
    name: Ident,
    data: syn::DataEnum,
}

impl ConfigEnumAnalyzer {
    pub fn new(name: Ident, data: syn::DataEnum) -> Self {
        Self { name, data }
    }

    fn serde_enum_ident(&self) -> Ident {
        Ident::new(&format!("{}Serde", self.name), self.name.span())
    }

    fn gen_serde_enum(&self) -> TokenStream {
        let enum_name = self.serde_enum_ident();
        let data = &self.data.variants;

        quote! {
            #[derive(serde::Serialize, serde::Deserialize)]
            enum #enum_name {
                #data
            }

        }
    }

    fn gen_variant_field(&self, variant: &Variant) -> (TokenStream, TokenStream) {
        let gen_fields_unnamed = |num: usize| {
            let mut input = Vec::new();
            let mut output = Vec::new();

            for i in 0..num {
                let arg_name = Ident::new(&format!("arg_{i}"), self.name.span());

                input.push(quote! { #arg_name });
                output.push(quote! { #arg_name.clone() });
            }

            (quote! (( #(#input),* )), quote! (( #(#output),* )))
        };
        let gen_fields_named = |fields: &FieldsNamed| {
            let mut input = Vec::new();
            let mut output = Vec::new();

            fields.named.iter().for_each(|field| {
                let ident = &field.ident;

                input.push(quote! {
                    #ident
                });
                output.push(quote! {
                    #ident: #ident.clone()
                });
            });

            (quote! {{ #(#input),* }}, quote! {{ #(#output),* }})
        };

        match &variant.fields {
            syn::Fields::Named(fields) => gen_fields_named(fields),
            syn::Fields::Unnamed(_) => gen_fields_unnamed(variant.fields.len()),
            syn::Fields::Unit => (quote! {}, quote! {}),
        }
    }

    fn gen_serialize_fn(&self) -> TokenStream {
        let enum_name = self.serde_enum_ident();
        let variants = self.data.variants.iter().map(|variant| {
            let variant_name = &variant.ident;
            let (variant_input, variant_output) = self.gen_variant_field(variant);

            quote! { Self::#variant_name #variant_input => #enum_name::#variant_name #variant_output }
        });
        let name = &self.name;

        quote! {
            impl serde::Serialize for #name {
                fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
                where
                    S: serde::Serializer {
                    let serde_state = match self {
                        #(#variants),*
                    };
                    serde_state.serialize(serializer)
                }
            }

        }
    }

    fn gen_deserialize_fn(&self) -> TokenStream {
        let enum_name = self.serde_enum_ident();
        let variants = self.data.variants.iter().map(|variant| {
            let variant_name = &variant.ident;
            let (variant_input, variant_output) = self.gen_variant_field(variant);

            quote! { #enum_name::#variant_name #variant_input => Self::#variant_name #variant_output }
        });
        let name = &self.name;

        quote! {
            impl<'de> serde::Deserialize<'de> for #name {
                fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
                where
                    D: serde::Deserializer<'de> {
                    let serde_state = #enum_name::deserialize(deserializer)?;
                    Ok(match serde_state {
                        #(#variants),*
                    })
                }
            }

        }
    }
}

impl ConfigAnalyzer for ConfigEnumAnalyzer {
    fn gen_serde_impl(&self) -> TokenStream {
        let struct_gen = self.gen_serde_enum();
        let serialize_gen = self.gen_serialize_fn();
        let deserialize_gen = self.gen_deserialize_fn();

        quote! {
            #struct_gen
            #serialize_gen
            #deserialize_gen
        }
    }

    fn gen_clone_impl(&self) -> TokenStream {
        let variants = self.data.variants.iter().map(|variant| {
            let variant_name = &variant.ident;
            let (variant_input, variant_output) = self.gen_variant_field(variant);

            quote! { Self::#variant_name #variant_input => Self::#variant_name #variant_output }
        });
        let name = &self.name;

        quote! {
            impl Clone for #name {
                fn clone(&self) -> Self {
                    match self {
                        #(#variants),*
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
