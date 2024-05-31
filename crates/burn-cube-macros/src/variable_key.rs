#[derive(Hash, PartialEq, Eq, Debug, Clone)]
pub(crate) enum VariableKey {
    LocalKey(String),
    Attribute((String, String)),
}

impl From<&syn::Ident> for VariableKey {
    fn from(value: &syn::Ident) -> Self {
        VariableKey::LocalKey(value.to_string())
    }
}

impl From<(&syn::Ident, &syn::Ident)> for VariableKey {
    fn from(value: (&syn::Ident, &syn::Ident)) -> Self {
        VariableKey::Attribute((value.0.to_string(), value.1.to_string()))
    }
}

impl From<String> for VariableKey {
    fn from(value: String) -> Self {
        VariableKey::LocalKey(value)
    }
}

impl From<(String, String)> for VariableKey {
    fn from(value: (String, String)) -> Self {
        VariableKey::Attribute(value)
    }
}
