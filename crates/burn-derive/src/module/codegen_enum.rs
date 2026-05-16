use super::{codegen::ModuleCodegen, record_enum::EnumModuleRecordCodegen};
use crate::module::{
    codegen_struct::{ModuleFieldType, parse_module_field_type},
    generics::{ModuleGenerics, parse_module_generics},
};
use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;
use syn::Visibility;

pub(crate) struct EnumModuleCodegen {
    pub name: Ident,
    pub variants: Vec<EnumVariant>,
    pub vis: Visibility,
    pub generics: ModuleGenerics,
}

impl ModuleCodegen for EnumModuleCodegen {
    type RecordCodegen = EnumModuleRecordCodegen;

    fn gen_num_params(&self) -> TokenStream {
        let match_body = self.gen_variants_match_fn(|_| {
            quote! {
                burn::module::Module::num_params(module)
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
            fn visit<Visitor: burn::module::ModuleVisitor>(&self, visitor: &mut Visitor) {
                #match_body
            }
        }
    }

    fn gen_collect_devices(&self) -> TokenStream {
        let match_body = self.gen_variants_match_fn(|_| {
            quote! {
                burn::module::Module::collect_devices(module, devices)
            }
        });

        quote! {
            fn collect_devices(
                &self,
                devices: burn::module::Devices
            ) -> burn::module::Devices {
                #match_body
            }
        }
    }

    fn gen_to_device(&self) -> TokenStream {
        let match_body = self.gen_variants_match_fn(|variant| {
            quote! {
                Self::#variant(burn::module::Module::to_device(module, device))
            }
        });

        quote! {
            fn to_device(self, device: &burn::tensor::Device) -> Self {
                #match_body
            }
        }
    }

    fn gen_fork(&self) -> TokenStream {
        let match_body = self.gen_variants_match_fn(|variant| {
            quote! {
                Self::#variant(burn::module::Module::fork(module, device))
            }
        });

        quote! {
            fn fork(self, device: &burn::tensor::Device) -> Self {
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
                    let result = burn::module::Module::map(module, mapper);
                    mapper.exit_module(#variant_str, #container_type);
                    Self::#variant(result)
                }
            }
        });

        quote! {
            fn map<Mapper: burn::module::ModuleMapper>(self, mapper: &mut Mapper) -> Self {
                #match_body
            }
        }
    }

    fn gen_valid(&self) -> TokenStream {
        let match_body = self.gen_variants_match_fn(|variant| {
            quote! {
                Self::#variant(burn::module::AutodiffModule::valid(module))
            }
        });

        quote! {
            fn valid(&self) -> Self {
                #match_body
            }
        }
    }

    fn gen_from_inner(&self) -> TokenStream {
        let match_body = self.gen_variants_match_fn_param("module", "Self::", |variant| {
            quote! {
                Self::#variant(burn::module::AutodiffModule::from_inner(module))
            }
        });

        quote! {
            fn from_inner(module: Self) -> Self {
                #match_body
            }
        }
    }

    fn gen_into_record(&self) -> TokenStream {
        let match_body = self.gen_variants_match_fn(|variant| {
            quote! {
                Self::Record::#variant(burn::module::Module::into_record(module))
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
                    Self::#variant(burn::module::Module::load_record(module, r))
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

    fn module_generics(&self) -> &ModuleGenerics {
        &self.generics
    }

    fn gen_display(&self) -> TokenStream {
        // Only tuple enum variants with exactly one field are supported
        let variant_prints = self.variants.iter().map(|variant| {
            let variant_name = &variant.ident;
            let field_names =
                (0..1).map(|i| syn::Ident::new(&format!("_{i}"), proc_macro2::Span::call_site()));

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
        });
        quote! {
            fn content(&self, mut content: burn::module::Content) -> Option<burn::module::Content> {
                match self {
                    #(#variant_prints)*
                }
            }
        }
    }
}

impl EnumModuleCodegen {
    pub fn from_ast(ast: &syn::DeriveInput) -> syn::Result<Self> {
        let mut generics = parse_module_generics(&ast.generics);
        Ok(Self {
            name: ast.ident.clone(),
            variants: parse_variants(ast, &mut generics)?,
            vis: ast.vis.clone(),
            generics,
        })
    }

    /// Generate the enum variants' match arms with the provided function
    fn gen_variants_match_fn<F>(&self, func: F) -> TokenStream
    where
        F: Fn(Ident) -> TokenStream,
    {
        self.gen_variants_match_fn_param("self", "Self::", func)
    }

    /// Generate a match expression over the given argument (e.g., `self`)
    /// and using the provided prefix for variants (e.g., `Self::`)
    fn gen_variants_match_fn_param<F>(&self, arg: &str, prefix: &str, func: F) -> TokenStream
    where
        F: Fn(Ident) -> TokenStream,
    {
        let match_arms = self.variants.iter().map(|variant| {
            let name = &variant.ident;
            let full_variant = syn::parse_str::<syn::Path>(&format!("{prefix}{name}")).unwrap();
            let arm_pattern = quote! { #full_variant(module) };
            let arm_code = func(name.clone());
            quote! { #arm_pattern => #arm_code, }
        });

        let arg = Ident::new(arg, Span::call_site());

        quote! {
            match #arg {
                #(#match_arms)*
            }
        }
    }

    /// Returns true if any field in any variant is considered a module.
    pub fn has_module_fields(&self) -> bool {
        self.variants
            .iter()
            .any(|variant| variant.field_type.is_module)
    }
}

/// Module enum variant
pub(crate) struct EnumVariant {
    pub ident: syn::Ident,
    pub ty: syn::Type,
    pub field_type: ModuleFieldType,
}

pub(crate) fn parse_variants(
    ast: &syn::DeriveInput,
    generics: &mut ModuleGenerics,
) -> syn::Result<Vec<EnumVariant>> {
    let enum_data = match &ast.data {
        syn::Data::Enum(data) => data,
        _ => return Err(syn::Error::new_spanned(ast, "Only enums are supported")),
    };

    let mut variants = Vec::new();

    for variant in enum_data.variants.iter() {
        for attr in &variant.attrs {
            if attr.path().is_ident("module") {
                Err(syn::Error::new_spanned(
                    variant,
                    "Module attributes are not supported for enum variants.",
                ))?;
            }
        }

        match &variant.fields {
            syn::Fields::Unnamed(fields) if fields.unnamed.len() == 1 => {
                let field = &fields.unnamed[0];

                // USE THE SAME PARSER AS STRUCTS
                // This gives us the is_module, is_param, etc. logic
                let field_type = parse_module_field_type(field, generics)?;

                variants.push(EnumVariant {
                    ident: variant.ident.clone(),
                    ty: field.ty.clone(),
                    field_type,
                });
            }
            syn::Fields::Unnamed(_) => {
                return Err(syn::Error::new_spanned(
                    variant,
                    "Module derive only supports tuple enum variants with exactly one field.",
                ));
            }
            syn::Fields::Named(_) => {
                return Err(syn::Error::new_spanned(
                    variant,
                    "Module derive does not support struct enum variants.",
                ));
            }
            syn::Fields::Unit => {
                return Err(syn::Error::new_spanned(
                    variant,
                    "Module derive does not support unit enum variants.",
                ));
            }
        }
    }

    Ok(variants)
}
