use super::codegen::ModuleCodegen;
use crate::module::codegen_struct::{ModuleField, ModuleFieldType, parse_module_field_type};
use proc_macro2::{Ident, Span, TokenStream};
use quote::{ToTokens, format_ident, quote};
use syn::Member;

pub(crate) struct EnumModuleCodegen {
    pub name: Ident,
    pub variants: Vec<EnumVariant>,
}

impl ModuleCodegen for EnumModuleCodegen {
    fn gen_num_params(&self) -> TokenStream {
        let match_body = self.gen_variants_match_fn(|variant| {
            let counts: Vec<_> = variant
                .module_fields()
                .map(|(binding, _)| quote! { burn::module::Module::num_params(#binding) })
                .collect();

            if counts.is_empty() {
                quote! { 0 }
            } else {
                quote! { #(#counts)+* }
            }
        });

        quote! {
            fn num_params(&self) -> usize {
                #match_body
            }
        }
    }

    fn gen_visit(&self) -> TokenStream {
        let container_type = self.container_type();
        let match_body = self.gen_variants_match_fn(|variant| {
            let visits: Vec<_> = variant
                .module_fields()
                .map(
                    |(binding, field)| match variant.field_scope(&self.name, field) {
                        Some((name, container)) => quote! {
                            visitor.enter_module(#name, #container);
                            burn::module::Module::visit(#binding, visitor);
                            visitor.exit_module(#name, #container);
                        },
                        None => quote! { burn::module::Module::visit(#binding, visitor); },
                    },
                )
                .collect();

            if visits.is_empty() {
                return quote! { {} };
            }

            let variant_str = variant.ident.to_string();
            quote! {
                {
                    visitor.enter_module(#variant_str, #container_type);
                    #(#visits)*
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
        let match_body = self.gen_variants_match_fn(|variant| {
            variant
                .module_fields()
                .fold(quote! { devices }, |devices, (binding, _)| {
                    quote! { burn::module::Module::collect_devices(#binding, #devices) }
                })
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
        let match_body = self.gen_rebuild_match_fn(|binding, field_type| {
            if field_type.is_module {
                quote! { burn::module::Module::to_device(#binding, device) }
            } else {
                quote! { #binding }
            }
        });

        quote! {
            fn to_device(self, device: &burn::tensor::Device) -> Self {
                #match_body
            }
        }
    }

    fn gen_fork(&self) -> TokenStream {
        let match_body = self.gen_rebuild_match_fn(|binding, field_type| {
            if field_type.is_module {
                quote! { burn::module::Module::fork(#binding, device) }
            } else {
                quote! { #binding }
            }
        });

        quote! {
            fn fork(self, device: &burn::tensor::Device) -> Self {
                #match_body
            }
        }
    }

    fn gen_map(&self) -> TokenStream {
        let container_type = self.container_type();
        let match_body = self.gen_variants_match_fn(|variant| {
            let rebuilt =
                variant.construct(variant.fields().iter().enumerate().map(|(index, field)| {
                    let binding = variant.binding(index);
                    if !field.field_type.is_module {
                        return quote! { #binding };
                    }
                    match variant.field_scope(&self.name, field) {
                        Some((name, container)) => quote! {
                            {
                                mapper.enter_module(#name, #container);
                                let result = burn::module::Module::map(#binding, mapper);
                                mapper.exit_module(#name, #container);
                                result
                            }
                        },
                        None => quote! { burn::module::Module::map(#binding, mapper) },
                    }
                }));

            if !variant.has_module_fields() {
                return rebuilt;
            }

            let variant_str = variant.ident.to_string();
            quote! {
                {
                    mapper.enter_module(#variant_str, #container_type);
                    let result = #rebuilt;
                    mapper.exit_module(#variant_str, #container_type);
                    result
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
        let match_body = self.gen_rebuild_match_fn(|binding, field_type| {
            if field_type.is_module {
                quote! { burn::module::Module::valid(#binding) }
            } else {
                quote! { #binding.clone() }
            }
        });

        quote! {
            fn valid(&self) -> Self {
                #match_body
            }
        }
    }

    fn gen_train(&self) -> TokenStream {
        let match_body = self.gen_rebuild_match_fn(|binding, field_type| {
            if field_type.is_module {
                quote! { burn::module::Module::train(#binding) }
            } else {
                quote! { #binding }
            }
        });

        quote! {
            fn train(self) -> Self {
                #match_body
            }
        }
    }

    fn gen_materialize(&self) -> TokenStream {
        let match_body = self.gen_rebuild_match_fn(|binding, field_type| {
            if field_type.is_module {
                quote! { burn::module::Module::materialize(#binding) }
            } else {
                quote! { #binding }
            }
        });

        quote! {
            fn materialize(self) -> Self {
                #match_body
            }
        }
    }

    fn gen_clone(&self) -> TokenStream {
        let match_body = self.gen_rebuild_match_fn(|binding, _| quote! { #binding.clone() });

        quote! {
            fn clone(&self) -> Self {
                #match_body
            }
        }
    }

    fn fields(&self) -> Vec<(&syn::Field, &ModuleFieldType)> {
        self.variants
            .iter()
            .flat_map(|variant| variant.fields())
            .map(|field| (&field.field, &field.field_type))
            .collect()
    }

    fn gen_display(&self) -> TokenStream {
        // Tuple enum variants have exactly one field, named enum variants at least one
        let variant_prints = self.variants.iter().map(|variant| {
            let variant_name = &variant.ident;
            let pattern = variant.pattern();
            let field_prints = variant.fields().iter().enumerate().map(|(index, field)| {
                let binding = variant.binding(index);
                let label = variant.display_label(field);
                if field.field_type.is_module {
                    quote! { .add(#label, #binding) }
                } else {
                    quote! { .add_debug_attribute(#label, #binding) }
                }
            });
            quote! {
                #pattern => {
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
        Ok(Self {
            name: ast.ident.clone(),
            variants: parse_variants(ast)?,
        })
    }

    /// Generate the enum variants' match arms with the provided function
    fn gen_variants_match_fn<F>(&self, func: F) -> TokenStream
    where
        F: Fn(&EnumVariant) -> TokenStream,
    {
        let match_arms = self.variants.iter().map(|variant| {
            let pattern = variant.pattern();
            let arm_code = func(variant);
            quote! { #pattern => #arm_code, }
        });

        quote! {
            match self {
                #(#match_arms)*
            }
        }
    }

    fn container_type(&self) -> String {
        format!("Enum:{}", self.name)
    }

    fn gen_rebuild_match_fn<F>(&self, field_expr: F) -> TokenStream
    where
        F: Fn(&Ident, &ModuleFieldType) -> TokenStream,
    {
        self.gen_variants_match_fn(|variant| {
            variant.construct(
                variant
                    .fields()
                    .iter()
                    .enumerate()
                    .map(|(index, field)| field_expr(&variant.binding(index), &field.field_type)),
            )
        })
    }

    /// Returns true if any field in any variant is considered a module.
    pub fn has_module_fields(&self) -> bool {
        self.variants.iter().any(EnumVariant::has_module_fields)
    }
}

/// Module enum variant
pub(crate) struct EnumVariant {
    pub ident: syn::Ident,
    pub fields: VariantFields,
}

pub(crate) enum VariantFields {
    Single(Box<ModuleField>),
    Named(Vec<ModuleField>),
}

impl EnumVariant {
    fn fields(&self) -> &[ModuleField] {
        match &self.fields {
            VariantFields::Single(field) => std::slice::from_ref(field.as_ref()),
            VariantFields::Named(fields) => fields,
        }
    }

    fn has_module_fields(&self) -> bool {
        self.fields().iter().any(|field| field.field_type.is_module)
    }

    fn module_fields(&self) -> impl Iterator<Item = (Ident, &ModuleField)> {
        self.fields()
            .iter()
            .enumerate()
            .filter(|(_, field)| field.field_type.is_module)
            .map(|(index, field)| (self.binding(index), field))
    }

    /// Named fields get a generated name so that a field called `device` or `mapper` cannot
    /// shadow the parameters of the generated methods.
    fn binding(&self, index: usize) -> Ident {
        match &self.fields {
            VariantFields::Single(_) => Ident::new("module", Span::call_site()),
            VariantFields::Named(_) => format_ident!("__burn_field_{index}"),
        }
    }

    fn pattern(&self) -> TokenStream {
        let name = &self.ident;
        match &self.fields {
            VariantFields::Single(_) => quote! { Self::#name(module) },
            VariantFields::Named(fields) => {
                let members = fields.iter().map(|field| &field.member);
                let bindings = (0..fields.len()).map(|index| self.binding(index));
                quote! { Self::#name { #(#members: #bindings),* } }
            }
        }
    }

    fn construct(&self, values: impl Iterator<Item = TokenStream>) -> TokenStream {
        let name = &self.ident;
        match &self.fields {
            VariantFields::Single(_) => quote! { Self::#name(#(#values)*) },
            VariantFields::Named(fields) => {
                let members = fields.iter().map(|field| &field.member);
                quote! { Self::#name { #(#members: #values),* } }
            }
        }
    }

    fn field_scope(&self, enum_name: &Ident, field: &ModuleField) -> Option<(String, String)> {
        match &self.fields {
            VariantFields::Single(_) => None,
            VariantFields::Named(_) => Some((
                field.member.to_token_stream().to_string(),
                format!("Variant:{enum_name}::{}", self.ident),
            )),
        }
    }

    fn display_label(&self, field: &ModuleField) -> String {
        match &self.fields {
            VariantFields::Single(_) => "_0".to_string(),
            VariantFields::Named(_) => field.member.to_token_stream().to_string(),
        }
    }
}

pub(crate) fn parse_variants(ast: &syn::DeriveInput) -> syn::Result<Vec<EnumVariant>> {
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

        let fields = match &variant.fields {
            syn::Fields::Unnamed(fields) if fields.unnamed.len() == 1 => {
                let field = &fields.unnamed[0];

                // USE THE SAME PARSER AS STRUCTS
                // This gives us the is_module, is_param, etc. logic
                VariantFields::Single(Box::new(ModuleField::new(
                    field.clone(),
                    Member::Unnamed(0.into()),
                    parse_module_field_type(field)?,
                )))
            }
            syn::Fields::Unnamed(_) => {
                return Err(syn::Error::new_spanned(
                    variant,
                    "Module derive only supports tuple enum variants with exactly one field.",
                ));
            }
            syn::Fields::Named(fields) if fields.named.is_empty() => {
                return Err(syn::Error::new_spanned(
                    variant,
                    "Module derive does not support struct enum variants without fields.",
                ));
            }
            syn::Fields::Named(fields) => VariantFields::Named(
                fields
                    .named
                    .iter()
                    .map(|field| {
                        let ident = field
                            .ident
                            .clone()
                            .expect("named fields have an identifier");
                        Ok(ModuleField::new(
                            field.clone(),
                            Member::Named(ident),
                            parse_module_field_type(field)?,
                        ))
                    })
                    .collect::<syn::Result<_>>()?,
            ),
            syn::Fields::Unit => {
                return Err(syn::Error::new_spanned(
                    variant,
                    "Module derive does not support unit enum variants.",
                ));
            }
        };

        variants.push(EnumVariant {
            ident: variant.ident.clone(),
            fields,
        });
    }

    Ok(variants)
}

#[cfg(test)]
mod tests {
    use super::parse_variants;
    use proc_macro2::TokenStream;
    use quote::quote;
    use syn::parse_quote;

    fn rejection_message(variant: TokenStream) -> String {
        let ast = parse_quote! { enum Net { #variant } };
        parse_variants(&ast)
            .err()
            .expect("the variant is rejected")
            .to_string()
    }

    #[test]
    fn named_variant_without_fields_is_rejected() {
        assert_eq!(
            rejection_message(quote! { Empty {} }),
            "Module derive does not support struct enum variants without fields."
        );
    }

    #[test]
    fn unit_variant_remains_rejected() {
        assert_eq!(
            rejection_message(quote! { Unit }),
            "Module derive does not support unit enum variants."
        );
    }

    #[test]
    fn tuple_variant_with_several_fields_remains_rejected() {
        assert_eq!(
            rejection_message(quote! { Pair(Linear, Linear) }),
            "Module derive only supports tuple enum variants with exactly one field."
        );
    }
}
