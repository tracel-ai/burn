use syn::{Attribute, Meta};

pub struct AttributeAnalyzer {
    attr: Attribute,
}

#[derive(Clone)]
pub struct AttributeItem {
    pub value: syn::Lit,
}

impl AttributeAnalyzer {
    pub fn new(attr: Attribute) -> Self {
        Self { attr }
    }

    pub fn item(&self) -> AttributeItem {
        let value = match &self.attr.meta {
            Meta::List(val) => val.parse_args::<syn::MetaNameValue>().unwrap(),
            Meta::NameValue(meta) => meta.clone(),
            Meta::Path(_) => panic!("Path meta unsupported"),
        };

        let lit = match value.value {
            syn::Expr::Lit(lit) => lit.lit,
            _ => panic!("Only literal is supported"),
        };

        AttributeItem { value: lit }
    }

    pub fn has_name(&self, name: &str) -> bool {
        Self::path_syn_name(self.attr.path()) == name
    }

    fn path_syn_name(path: &syn::Path) -> String {
        let length = path.segments.len();
        let mut name = String::new();
        for (i, segment) in path.segments.iter().enumerate() {
            if i == length - 1 {
                name += segment.ident.to_string().as_str();
            } else {
                let tmp = segment.ident.to_string() + "::";
                name += tmp.as_str();
            }
        }
        name
    }
}
