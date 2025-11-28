use super::{codegen::ModuleCodegen, record_enum::EnumModuleRecordCodegen};
use crate::shared::enum_variant::{EnumVariant, parse_variants};
use proc_macro2::{Ident, TokenStream};
use quote::quote;
use syn::Visibility;

pub(crate) struct EnumModuleCodegen {
    pub name: Ident,
    pub variants: Vec<EnumVariant>,
    pub vis: Visibility,
}

impl ModuleCodegen for EnumModuleCodegen {
    type RecordCodegen = EnumModuleRecordCodegen;

    fn gen_num_params(&self) -> TokenStream {
        let match_body = self.gen_variants_match_fn(|_| {
            quote! {
                burn::module::Module::<B>::num_params(module)
            }
        });

        quote! {
            fn num_params(&self) -> usize {
                #match_body
            }
        }
    }

    fn gen_visit(&self) -> TokenStream {
        let enum_name = self.name.to_string();
        let container_type = format!("Enum:{}", enum_name);
        let match_body = self.gen_variants_match_fn(|variant_name| {
            let variant_str = variant_name.to_string();
            quote! {
                {
                    visitor.enter_module(#variant_str, #container_type);
                    burn::module::Module::visit(module, visitor);
                    visitor.exit_module(#variant_str, #container_type);
                }
            }
        });

        quote! {
            fn visit<Visitor: burn::module::ModuleVisitor<B>>(&self, visitor: &mut Visitor) {
                #match_body
            }
        }
    }

    fn gen_collect_devices(&self) -> TokenStream {
        let match_body = self.gen_variants_match_fn(|_| {
            quote! {
                burn::module::Module::<B>::collect_devices(module, devices)
            }
        });

        quote! {
            fn collect_devices(
                &self,
                devices: burn::module::Devices<B>
            ) -> burn::module::Devices<B> {
                #match_body
            }
        }
    }

    fn gen_to_device(&self) -> TokenStream {
        let match_body = self.gen_variants_match_fn(|variant| {
            quote! {
                Self::#variant(burn::module::Module::<B>::to_device(module, device))
            }
        });

        quote! {
            fn to_device(self, device: &B::Device) -> Self {
                #match_body
            }
        }
    }

    fn gen_fork(&self) -> TokenStream {
        let match_body = self.gen_variants_match_fn(|variant| {
            quote! {
                Self::#variant(burn::module::Module::<B>::fork(module, device))
            }
        });

        quote! {
            fn fork(self, device: &B::Device) -> Self {
                #match_body
            }
        }
    }

    fn gen_map(&self) -> TokenStream {
        let enum_name = self.name.to_string();
        let container_type = format!("Enum:{}", enum_name);
        let match_body = self.gen_variants_match_fn(|variant| {
            let variant_str = variant.to_string();
            quote! {
                {
                    mapper.enter_module(#variant_str, #container_type);
                    let result = burn::module::Module::<B>::map(module, mapper);
                    mapper.exit_module(#variant_str, #container_type);
                    Self::#variant(result)
                }
            }
        });

        quote! {
            fn map<Mapper: burn::module::ModuleMapper<B>>(self, mapper: &mut Mapper) -> Self {
                #match_body
            }
        }
    }

    fn gen_valid(&self) -> TokenStream {
        let match_body = self.gen_variants_match_fn(|variant| {
            quote! {
                Self::InnerModule::#variant(burn::module::AutodiffModule::<B>::valid(module))
            }
        });

        quote! {
            fn valid(&self) -> Self::InnerModule {
                #match_body
            }
        }
    }

    fn gen_into_record(&self) -> TokenStream {
        let match_body = self.gen_variants_match_fn(|variant| {
            quote! {
                Self::Record::#variant(burn::module::Module::<B>::into_record(module))
            }
        });

        quote! {
            fn into_record(self) -> Self::Record {
                #match_body
            }
        }
    }

    fn gen_load_record(&self) -> TokenStream {
        let match_body = self.gen_variants_match_fn(|variant| {
            quote! {
                {
                    let Self::Record::#variant(r) = record else {panic!("Can't parse record from a different variant");};
                    Self::#variant(burn::module::Module::<B>::load_record(module, r))
                }
            }
        });

        quote! {
            fn load_record(self, record: Self::Record) -> Self {
                #match_body
            }
        }
    }

    fn gen_clone(&self) -> TokenStream {
        let match_body = self.gen_variants_match_fn(|variant| {
            quote! {
                Self::#variant(module.clone())
            }
        });

        quote! {
            fn clone(&self) -> Self {
                #match_body
            }
        }
    }

    fn record_codegen(self) -> Self::RecordCodegen {
        EnumModuleRecordCodegen::new(self.variants, self.vis)
    }
}

impl EnumModuleCodegen {
    pub fn from_ast(ast: &syn::DeriveInput) -> Self {
        Self {
            name: ast.ident.clone(),
            variants: parse_variants(ast),
            vis: ast.vis.clone(),
        }
    }

    /// Generate the enum variants' match arm with the provided function
    fn gen_variants_match_fn<F>(&self, func: F) -> TokenStream
    where
        F: Fn(Ident) -> TokenStream,
    {
        let mut match_arms = quote! {};

        for variant in self.variants.iter() {
            let name = &variant.ident;
            let arm_pattern = quote! {Self::#name(module)};
            let arm_code = func(name.clone());

            match_arms.extend(quote! {#arm_pattern => #arm_code,})
        }

        quote! {
            match self {
                #match_arms
            }
        }
    }
}
