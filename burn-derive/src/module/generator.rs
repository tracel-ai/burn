use crate::shared::field::{parse_fields, FieldTypeAnalyzer};
use proc_macro2::{Ident, TokenStream};
use quote::quote;

pub struct FnGenerator {
    pub fields: Vec<FieldTypeAnalyzer>,
}

impl FnGenerator {
    pub fn from_ast(ast: &syn::DeriveInput) -> Self {
        Self {
            fields: parse_fields(ast)
                .into_iter()
                .map(FieldTypeAnalyzer::new)
                .collect(),
        }
    }

    pub fn gen_num_params_fn(&self) -> TokenStream {
        let body = self.gen_fields_fn(|name| {
            quote! {
                num_params += burn::module::Module::<B>::num_params(&self.#name);
            }
        });

        quote! {
            fn num_params(&self) -> usize {
                let mut num_params = 0;
                #body
                num_params
            }
        }
    }

    pub fn gen_into_record_fn(&self) -> TokenStream {
        let body = self.gen_fields_fn(|name| {
            quote! {
                #name: burn::module::Module::<B>::into_record(self.#name),
            }
        });

        quote! {
            fn into_record(self) -> Self::Record {
                Self::Record {
                    #body
                }
            }
        }
    }

    pub fn gen_load_record_fn(&self) -> TokenStream {
        let body = self.gen_fields_fn(|name| {
            quote! {
                #name: burn::module::Module::<B>::load_record(self.#name, record.#name),
            }
        });

        quote! {
            fn load_record(self, record: Self::Record) -> Self {
                Self {
                    #body
                }
            }
        }
    }

    pub fn gen_visit_fn(&self) -> TokenStream {
        let body = self.gen_fields_fn(|name| {
            quote! {
                burn::module::Module::visit(&self.#name, visitor);
            }
        });

        quote! {
            fn visit<V: burn::module::ModuleVisitor<B>>(&self, visitor: &mut V) {
                #body
            }
        }
    }

    pub fn gen_map_fn(&self) -> TokenStream {
        let (names, body) = self.gen_fields_fn_names(|name| {
            quote! {
                let #name = burn::module::Module::map(self.#name, mapper);
            }
        });

        quote! {
            fn map<M: burn::module::ModuleMapper<B>>(self, mapper: &mut M) -> Self {
                #body

                Self {
                    #(#names),*
                }
            }
        }
    }

    pub fn gen_devices_fn(&self) -> TokenStream {
        let body = self.gen_fields_fn(|name| {
            quote! {
                devices.append(&mut burn::module::Module::<B>::devices(&self.#name));
            }
        });

        quote! {
            fn devices(&self) -> Vec<B::Device> {
                let mut devices = Vec::new();
                #body
                devices
            }
        }
    }

    pub fn gen_to_device_fn(&self) -> TokenStream {
        let (names, body) = self.gen_fields_fn_names(|name| {
            quote! {
                let #name = burn::module::Module::<B>::to_device(self.#name, device);
            }
        });

        quote! {
            fn to_device(self, device: &B::Device) -> Self {
                #body

                Self {
                    #(#names),*
                }
            }
        }
    }

    pub fn gen_detach_fn(&self) -> TokenStream {
        let (names, body) = self.gen_fields_fn_names(|name| {
            quote! {
                let #name = burn::module::Module::<B>::detach(self.#name);
            }
        });

        quote! {
            fn detach(self) -> Self {
                #body

                Self {
                    #(#names),*
                }

            }
        }
    }

    pub fn gen_inner_fn(&self) -> TokenStream {
        let (names, body) = self.gen_fields_fn_names(|name| {
            quote! {
                let #name = burn::module::ADModule::<B>::inner(self.#name);
            }
        });

        quote! {
            fn inner(self) -> Self::InnerModule {
                #body

                Self::InnerModule {
                    #(#names),*
                }
            }
        }
    }

    pub fn gen_from_inner_fn(&self) -> TokenStream {
        let (names, body) = self.gen_fields_fn_names(|name| {
            quote! {
                let #name = burn::module::ADModule::<B>::from_inner(module.#name);
            }
        });

        quote! {
            fn from_inner(module: Self::InnerModule) -> Self {
                #body

                Self {
                    #(#names),*
                }
            }
        }
    }

    pub fn gen_clone_fn(&self) -> TokenStream {
        let (names, body) = self.gen_fields_fn_names(|name| {
            quote! {
                let #name = self.#name.clone();
            }
        });

        quote! {
            fn clone(&self) -> Self {
                #body

                Self {
                    #(#names),*
                }
            }
        }
    }

    pub fn gen_fields_fn_names<F>(&self, func: F) -> (Vec<Ident>, TokenStream)
    where
        F: Fn(Ident) -> TokenStream,
    {
        let mut body = quote! {};
        let mut names = Vec::new();

        for field in self.fields.iter() {
            let name = field.ident();

            names.push(name.clone());
            body.extend(func(field.ident()));
        }

        (names, body)
    }

    pub fn gen_fields_fn<F>(&self, func: F) -> TokenStream
    where
        F: Fn(Ident) -> TokenStream,
    {
        let mut body = quote! {};

        for field in self.fields.iter() {
            body.extend(func(field.ident()));
        }

        body
    }
}
