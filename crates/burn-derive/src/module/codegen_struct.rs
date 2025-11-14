use super::{codegen::ModuleCodegen, record_struct::StructModuleRecordCodegen};
use crate::shared::field::{FieldTypeAnalyzer, parse_fields};
use proc_macro2::{Ident, TokenStream};
use quote::quote;
use syn::Visibility;

pub(crate) struct StructModuleCodegen {
    pub name: Ident,
    pub fields: Vec<FieldTypeAnalyzer>,
    pub vis: Visibility,
}

impl ModuleCodegen for StructModuleCodegen {
    type RecordCodegen = StructModuleRecordCodegen;

    fn gen_num_params(&self) -> TokenStream {
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

    fn gen_visit(&self) -> TokenStream {
        let struct_name = self.name.to_string();
        let body = self.gen_fields_fn(|name| {
            let name_str = name.to_string();
            quote! {
                visitor.enter_module(#name_str, #struct_name);
                burn::module::Module::visit(&self.#name, visitor);
                visitor.exit_module(#name_str, #struct_name);
            }
        });

        quote! {
            fn visit<Visitor: burn::module::ModuleVisitor<B>>(&self, visitor: &mut Visitor) {
                #body
            }
        }
    }

    fn gen_collect_devices(&self) -> TokenStream {
        let body = self.gen_fields_fn(|name| {
            quote! {
                let devices = burn::module::Module::<B>::collect_devices(&self.#name, devices);
            }
        });

        quote! {
            fn collect_devices(
                &self,
                devices: burn::module::Devices<B>
            ) -> burn::module::Devices<B> {
                #body

                devices
            }
        }
    }

    fn gen_to_device(&self) -> TokenStream {
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

    fn gen_fork(&self) -> TokenStream {
        let (names, body) = self.gen_fields_fn_names(|name| {
            quote! {
                let #name = burn::module::Module::<B>::fork(self.#name, device);
            }
        });

        quote! {
            fn fork(self, device: &B::Device) -> Self {
                #body

                Self {
                    #(#names),*
                }
            }
        }
    }

    fn gen_map(&self) -> TokenStream {
        let struct_name = self.name.to_string();
        let (names, body) = self.gen_fields_fn_names(|name| {
            let name_str = name.to_string();
            quote! {
                mapper.enter_module(#name_str, #struct_name);
                let #name = burn::module::Module::<B>::map(self.#name, mapper);
                mapper.exit_module(#name_str, #struct_name);
            }
        });

        quote! {
            fn map<Mapper: burn::module::ModuleMapper<B>>(self, mapper: &mut Mapper) -> Self {
                #body

                Self {
                    #(#names),*
                }
            }
        }
    }

    fn gen_valid(&self) -> TokenStream {
        let (names, body) = self.gen_fields_fn_names(|name| {
            quote! {
                let #name = burn::module::AutodiffModule::<B>::valid(&self.#name);
            }
        });

        quote! {
            fn valid(&self) -> Self::InnerModule {
                #body

                Self::InnerModule {
                    #(#names),*
                }
            }
        }
    }

    fn gen_from_inner(&self) -> TokenStream {
        let (names, body) = self.gen_fields_fn_names(|name| {
            quote! {
                let #name = burn::module::AutodiffModule::<B>::from_inner(#name);
            }
        });

        // Destructure inner module to move all fields
        let destructure = quote! {
            let Self::InnerModule { #(#names),* } = module;
        };

        quote! {
            fn from_inner(module: Self::InnerModule) -> Self {
                #destructure
                #body

                Self {
                    #(#names),*
                }
            }
        }
    }

    fn gen_into_record(&self) -> TokenStream {
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

    fn gen_load_record(&self) -> TokenStream {
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

    fn gen_clone(&self) -> TokenStream {
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

    fn record_codegen(self) -> Self::RecordCodegen {
        StructModuleRecordCodegen::new(self.fields, self.vis)
    }
}

impl StructModuleCodegen {
    pub fn from_ast(ast: &syn::DeriveInput) -> Self {
        Self {
            name: ast.ident.clone(),
            fields: parse_fields(ast)
                .into_iter()
                .map(FieldTypeAnalyzer::new)
                .collect(),
            vis: ast.vis.clone(),
        }
    }

    fn gen_fields_fn_names<F>(&self, func: F) -> (Vec<Ident>, TokenStream)
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

    fn gen_fields_fn<F>(&self, func: F) -> TokenStream
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
