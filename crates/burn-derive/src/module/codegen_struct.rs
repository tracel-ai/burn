use std::collections::HashSet;

use crate::module::generics::{
    GenericKind, ModuleGenerics, parse_module_generics, parse_ty_generics,
};

use super::{codegen::ModuleCodegen, record_struct::StructModuleRecordCodegen};
use proc_macro2::{Ident, TokenStream};
use quote::{ToTokens, quote};
use syn::{Field, Visibility};

pub(crate) struct StructModuleCodegen {
    pub name: Ident,
    pub fields: Vec<ModuleField>,
    pub vis: Visibility,
    pub generics: ModuleGenerics,
}

impl ModuleCodegen for StructModuleCodegen {
    type RecordCodegen = StructModuleRecordCodegen;

    fn gen_num_params(&self) -> TokenStream {
        let body = self.gen_fields_fn(|name, field_type| {
            if field_type.is_parameter_module() || field_type.maybe_generic_module() {
                quote! {
                    num_params += burn::module::Module::<B>::num_params(&self.#name);
                }
            } else {
                quote! {} // other fields have 0 params
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
        let container_type = format!("Struct:{}", struct_name);
        let body = self.gen_fields_fn(|name, field_type| {
            if field_type.is_parameter_module() || field_type.maybe_generic_module() {
                let name_str = name.to_string();
                quote! {
                    visitor.enter_module(#name_str, #container_type);
                    burn::module::Module::visit(&self.#name, visitor);
                    visitor.exit_module(#name_str, #container_type);
                }
            } else {
                quote! {}
            }
        });

        quote! {
            fn visit<Visitor: burn::module::ModuleVisitor<B>>(&self, visitor: &mut Visitor) {
                #body
            }
        }
    }

    fn gen_collect_devices(&self) -> TokenStream {
        let body = self.gen_fields_fn(|name, field_type| {
            if field_type.is_module || field_type.maybe_generic_module() {
                quote! {
                    let devices = burn::module::Module::<B>::collect_devices(&self.#name, devices);
                }
            } else {
                quote! {}
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
        let (names, body) = self.gen_fields_fn_names(|name, field_type| {
            if field_type.is_module || field_type.maybe_generic_module() {
                quote! {
                    let #name = burn::module::Module::<B>::to_device(self.#name, device);
                }
            } else {
                quote! { let #name = self.#name; }
            }
        });

        quote! {
            fn to_device(self, device: &B::Device) -> Self {
                #body
                Self { #(#names),* }
            }
        }
    }

    fn gen_fork(&self) -> TokenStream {
        let (names, body) = self.gen_fields_fn_names(|name, field_type| {
            if field_type.is_module || field_type.maybe_generic_module() {
                quote! {
                    let #name = burn::module::Module::<B>::fork(self.#name, device);
                }
            } else {
                quote! { let #name = self.#name; }
            }
        });

        quote! {
            fn fork(self, device: &B::Device) -> Self {
                #body
                Self { #(#names),* }
            }
        }
    }

    fn gen_map(&self) -> TokenStream {
        let struct_name = self.name.to_string();
        let container_type = format!("Struct:{}", struct_name);
        let (names, body) = self.gen_fields_fn_names(|name, field_type| {
            if field_type.is_parameter_module() || field_type.maybe_generic_module() {
                let name_str = name.to_string();
                quote! {
                    mapper.enter_module(#name_str, #container_type);
                    let #name = burn::module::Module::<B>::map(self.#name, mapper);
                    mapper.exit_module(#name_str, #container_type);
                }
            } else {
                quote! { let #name = self.#name; }
            }
        });

        quote! {
            fn map<Mapper: burn::module::ModuleMapper<B>>(self, mapper: &mut Mapper) -> Self {
                #body
                Self { #(#names),* }
            }
        }
    }

    fn gen_valid(&self) -> TokenStream {
        let (names, body) = self.gen_fields_fn_names(|name, field_type| {
            if field_type.is_module || field_type.maybe_generic_module() {
                quote! {
                    let #name = burn::module::AutodiffModule::<B>::valid(&self.#name);
                }
            } else {
                quote! { let #name = self.#name.clone(); }
            }
        });

        quote! {
            fn valid(&self) -> Self::InnerModule {
                #body
                Self::InnerModule { #(#names),* }
            }
        }
    }

    fn gen_from_inner(&self) -> TokenStream {
        let (names, body) = self.gen_fields_fn_names(|name, field_type| {
            if field_type.is_module || field_type.maybe_generic_module() {
                quote! {
                    let #name = burn::module::AutodiffModule::<B>::from_inner(#name);
                }
            } else {
                quote! { let #name = #name; }
            }
        });

        let destructure = quote! {
            let Self::InnerModule { #(#names),* } = module;
        };

        quote! {
            fn from_inner(module: Self::InnerModule) -> Self {
                #destructure
                #body
                Self { #(#names),* }
            }
        }
    }

    fn gen_into_record(&self) -> TokenStream {
        let body = self.gen_fields_fn(|name, field_type| {
            if field_type.is_persistent_module() || field_type.maybe_generic_module() {
                quote! { #name: burn::module::Module::<B>::into_record(self.#name), }
            } else {
                match field_type.attr {
                    Some(ModuleFieldAttribute::Constant) => {
                        quote! { #name: burn::module::ValueRecord::new(self.#name), }
                    }
                    // Default (None) gets skipped
                    None | Some(ModuleFieldAttribute::Skip) => {
                        quote! { #name: burn::module::EmptyRecord::new(), }
                    }
                }
            }
        });

        quote! {
            fn into_record(self) -> Self::Record {
                Self::Record { #body }
            }
        }
    }

    fn gen_load_record(&self) -> TokenStream {
        let body = self.gen_fields_fn(|name, field_type| {
            if field_type.is_persistent_module() || field_type.maybe_generic_module() {
                quote! { #name: burn::module::Module::<B>::load_record(self.#name, record.#name), }
            } else {
                match field_type.attr {
                    Some(ModuleFieldAttribute::Constant) => {
                        quote! { #name: record.#name.consume(), }
                    }
                    // Default (None) gets skipped
                    None | Some(ModuleFieldAttribute::Skip) => {
                        quote! { #name: self.#name, }
                    }
                }
            }
        });

        quote! {
            fn load_record(self, record: Self::Record) -> Self {
                Self { #body }
            }
        }
    }

    fn gen_clone(&self) -> TokenStream {
        let (names, body) = self.gen_fields_fn_names(|name, _field_type| {
            quote! {
                let #name = self.#name.clone();
            }
        });

        quote! {
            fn clone(&self) -> Self {
                #body
                Self { #(#names),* }
            }
        }
    }

    fn record_codegen(self) -> Self::RecordCodegen {
        StructModuleRecordCodegen::new(self.fields, self.vis)
    }

    fn module_generics(&self) -> &ModuleGenerics {
        &self.generics
    }

    fn gen_display(&self) -> TokenStream {
        let struct_name = self.name.to_string();
        let field_prints = self.fields.iter().map(|field| {
            let field_name = field.ident();
            if field.field_type.is_module || field.field_type.maybe_generic_module() {
                // Standard module type, use underlying `ModuleDisplay` impl
                quote! { .add(stringify!(#field_name), &self.#field_name) }
            } else {
                // Not a module, use the debug implementation
                quote! {
                    .add_debug_attribute(stringify!(#field_name), &self.#field_name)
                }
            }
        });
        quote! {
            fn content(&self, mut content: burn::module::Content) -> Option<burn::module::Content> {
                content
                    .set_top_level_type(&stringify!(#struct_name))
                    #(#field_prints)*
                    .optional()
            }
        }
    }
}

impl StructModuleCodegen {
    pub fn from_ast(ast: &syn::DeriveInput) -> syn::Result<Self> {
        let mut generics = parse_module_generics(&ast.generics);
        Ok(Self {
            name: ast.ident.clone(),
            fields: parse_module_fields(ast, &mut generics)?,
            vis: ast.vis.clone(),
            generics,
        })
    }

    fn gen_fields_fn_names<F>(&self, func: F) -> (Vec<Ident>, TokenStream)
    where
        F: Fn(Ident, &ModuleFieldType) -> TokenStream,
    {
        let mut body = quote! {};
        let mut names = Vec::new();

        for field in self.fields.iter() {
            let name = field.ident();

            names.push(name.clone());
            body.extend(func(name, &field.field_type));
        }

        (names, body)
    }

    fn gen_fields_fn<F>(&self, func: F) -> TokenStream
    where
        F: Fn(Ident, &ModuleFieldType) -> TokenStream,
    {
        let mut body = quote! {};

        for field in self.fields.iter() {
            body.extend(func(field.ident(), &field.field_type));
        }

        body
    }
}

#[derive(new)]
pub struct ModuleField {
    pub field: Field,
    pub field_type: ModuleFieldType,
}

impl ModuleField {
    pub fn ident(&self) -> Ident {
        self.field.ident.clone().unwrap()
    }
}

#[derive(Debug)]
pub enum ModuleFieldAttribute {
    Constant,
    Skip,
}

#[derive(Default, Debug)]
pub struct ModuleFieldType {
    pub is_module: bool,
    pub attr: Option<ModuleFieldAttribute>,
    pub generic_idents: HashSet<Ident>,
}

impl ModuleFieldType {
    /// Returns true if the field is a module with parameters
    /// (i.e., a real module that is neither skipped nor constant).
    pub fn is_parameter_module(&self) -> bool {
        self.is_module && self.attr.is_none()
    }

    /// Returns true for modules that should be persisted, including constants.
    pub fn is_persistent_module(&self) -> bool {
        self.is_module && !matches!(self.attr, Some(ModuleFieldAttribute::Skip))
    }

    /// Returns true for generic fields that are assumed to be modules.
    pub fn maybe_generic_module(&self) -> bool {
        // We assumed it might be a module generic if the field is not marked
        // by any attributes (skip or constant)
        !self.generic_idents.is_empty() && self.attr.is_none()
    }
}

pub(crate) fn parse_module_fields(
    ast: &syn::DeriveInput,
    generics: &mut ModuleGenerics,
) -> syn::Result<Vec<ModuleField>> {
    let mut fields = Vec::new();

    match &ast.data {
        syn::Data::Struct(struct_data) => {
            for field in struct_data.fields.iter() {
                let field_type = parse_module_field_type(field, generics)?;
                fields.push(ModuleField::new(field.clone(), field_type));
            }
        }
        syn::Data::Enum(_) => panic!("Only struct can be derived"),
        syn::Data::Union(_) => panic!("Only struct can be derived"),
    };
    Ok(fields)
}

pub(crate) fn parse_module_field_type(
    field: &Field,
    generics: &mut ModuleGenerics,
) -> syn::Result<ModuleFieldType> {
    let mut field_type = ModuleFieldType::default();

    // Check for generics
    let mut has_backend = false;
    let mut has_module_bound = false;
    let field_generics = parse_ty_generics(&field.ty)
        .into_iter()
        .filter_map(|ident| {
            if ident == "B" {
                has_backend = true;
                None
            } else if generics.is_bounded_module(&ident) {
                has_module_bound = true;
                None
            } else {
                Some(ident)
            }
        })
        .collect::<HashSet<_>>();

    // Infer if a field is a module
    let is_primitive = is_primitive_type(&field.ty);
    let is_param = is_param_type(&field.ty);
    let is_tensor = is_tensor_type(&field.ty);

    for attr in &field.attrs {
        if attr.path().is_ident("module") {
            attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("constant") {
                    // Mark field attribute and generic
                    field_type.attr = Some(ModuleFieldAttribute::Constant);
                    for ty in &field_generics {
                        generics.update(ty, GenericKind::Constant);
                    }
                    Ok(())
                } else if meta.path.is_ident("skip") {
                    // Mark field attribute and generic
                    field_type.attr = Some(ModuleFieldAttribute::Skip);
                    for ty in &field_generics {
                        generics.update(ty, GenericKind::Skip);
                    }
                    Ok(())
                } else {
                    let path = meta.path.to_token_stream().to_string();
                    Err(meta.error(format!("Unsupported module attribute: {}", path)))
                }?;

                if is_param && field_type.attr.is_some() {
                    Err(meta.error("Fields of type 'Param' should not be marked as 'constant' or 'skip'. Use a 'Tensor' instead."))
                } else {
                    Ok(())
                }
            })?;
        }
    }

    field_type.is_module =
        !is_primitive && (has_module_bound || is_param || is_tensor || has_backend);
    field_type.generic_idents = field_generics;

    Ok(field_type)
}

fn type_matches_ident(ty: &syn::Type, idents: &[&str]) -> bool {
    if let syn::Type::Path(type_path) = ty {
        // Look at the last segment of the path (e.g., 'Param' in 'burn::module::Param')
        if let Some(segment) = type_path.path.segments.last() {
            return idents.contains(&segment.ident.to_string().as_str());
        }
    }
    false
}

fn is_primitive_type(ty: &syn::Type) -> bool {
    type_matches_ident(
        ty,
        &[
            "bool", "u8", "u16", "u32", "u64", "usize", "i8", "i16", "i32", "i64", "isize", "f32",
            "f64", "String",
        ],
    )
}

fn is_tensor_type(ty: &syn::Type) -> bool {
    type_matches_ident(ty, &["Tensor"])
}

fn is_param_type(ty: &syn::Type) -> bool {
    type_matches_ident(ty, &["Param"])
}
