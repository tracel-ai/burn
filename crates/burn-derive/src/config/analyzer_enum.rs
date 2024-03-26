use crate::shared::enum_variant::map_enum_variant;

use super::ConfigAnalyzer;
use proc_macro2::{Ident, TokenStream};
use quote::quote;

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
            #[derive(burn::serde::Serialize, burn::serde::Deserialize)]
            #[serde(crate = "burn::serde")]
            enum #enum_name {
                #data
            }

        }
    }

    fn gen_serialize_fn(&self) -> TokenStream {
        let enum_name = self.serde_enum_ident();
        let variants = self.data.variants.iter().map(|variant| {
            let variant_name = &variant.ident;
            let (inputs, outputs) = map_enum_variant(variant, |ident| quote! { #ident.clone() });

            quote! { Self::#variant_name #inputs => #enum_name::#variant_name #outputs }
        });

        let name = &self.name;

        quote! {
            impl burn::serde::Serialize for #name {
                fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
                where
                    S: burn::serde::Serializer {
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
            let (inputs, outputs) = map_enum_variant(variant, |ident| quote! { #ident.clone() });

            quote! { #enum_name::#variant_name #inputs => Self::#variant_name #outputs }
        });
        let name = &self.name;

        quote! {
            impl<'de> burn::serde::Deserialize<'de> for #name {
                fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
                where
                    D: burn::serde::Deserializer<'de> {
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
            let (inputs, outputs) = map_enum_variant(variant, |ident| quote! { #ident.clone() });

            quote! { Self::#variant_name #inputs => Self::#variant_name #outputs }
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
