use std::collections::HashMap;

#[derive(new, Hash, PartialEq, Eq, Debug, Clone)]
/// Identifies a variable uniquely
pub struct VariableIdent {
    name: String,
    repeat: u8,
    scope: u8,
    field: Option<String>,
}

#[derive(new, Eq, PartialEq, Hash, Debug)]
/// Identifies a variable, with possible collisions when variables are redeclared
struct VariableKey {
    name: String,
    scope: u8,
}

#[derive(Debug, Default)]
/// Tracks variable uses
pub(crate) struct VariableTracker {
    scopes_declared: HashMap<String, Vec<u8>>,
    analysis_repeats: HashMap<VariableKey, u8>,
    codegen_repeats: HashMap<VariableKey, u8>,
    variable_uses: HashMap<VariableIdent, VariableUse>,
    pub errors: Vec<syn::Error>,
}

#[derive(Debug, Default)]
/// Encapsulates number of uses and whether this implies cloning
pub(crate) struct VariableUse {
    pub num_used: usize,
    pub is_comptime: bool,
}

impl VariableUse {
    pub fn should_clone(&self) -> bool {
        self.num_used > 1
    }
}

impl VariableTracker {
    /// During analysis, tracks a variable declaration
    pub(crate) fn analyze_declare(&mut self, name: String, scope: u8, is_comptime: bool) {
        if let Some(scopes) = self.scopes_declared.get_mut(&name) {
            if !scopes.contains(&scope) {
                scopes.push(scope);
            }
        } else {
            self.scopes_declared.insert(name.clone(), vec![scope]);
        }

        let key = VariableKey::new(name.clone(), scope);
        let repeat = if let Some(count) = self.analysis_repeats.get_mut(&key) {
            *count += 1;
            *count
        } else {
            self.analysis_repeats.insert(key, 0);
            0
        };

        let analysis = VariableUse {
            num_used: 1,
            is_comptime,
        };
        let variable_ident = VariableIdent::new(name, repeat, scope, None);
        self.variable_uses.insert(variable_ident, analysis);
    }

    /// During analysis, tracks a variable use
    pub(crate) fn analyze_reuse(&mut self, ident: &syn::Ident, scope: u8, field: Option<String>) {
        let name = ident.to_string();
        let scopes_declared = match self.scopes_declared.get(&name) {
            Some(val) => val,
            None => {
                self.errors
                    .push(syn::Error::new_spanned(ident, "Variable not declared"));
                return;
            }
        };

        let scope = *scopes_declared
            .iter()
            .filter(|s| **s <= scope)
            .max()
            .unwrap();
        let key = VariableKey::new(name.clone(), scope);

        // If the name and scope do not match a declared variable,
        // then we are using a variable declared in a parent scope, and
        // cloning must always happen, therefore no need for further analysis
        if let Some(repeat) = self.analysis_repeats.get(&key) {
            let variable = VariableIdent::new(name, *repeat, scope, field);
            self.analyze(&variable);
        }
    }

    /// Increments variable use and its parent struct if need be
    fn analyze(&mut self, variable_ident: &VariableIdent) {
        match self.variable_uses.get_mut(variable_ident) {
            Some(variable_use) => {
                variable_use.num_used += 1;
            }
            None => {
                // If variable was not inserted yet, it must be a field
                if variable_ident.field.is_some() {
                    let mut parent_ident = variable_ident.clone();
                    parent_ident.field = None;
                    let parent = self.variable_uses.get(&parent_ident).unwrap();

                    let attr_analysis = VariableUse {
                        num_used: 1,
                        is_comptime: parent.is_comptime,
                    };
                    self.variable_uses
                        .insert(variable_ident.clone(), attr_analysis);
                } else {
                    panic!("Variable not declared");
                }
            }
        };

        // Whether a field was previously seen or not, we must increase the use of the parent struct
        if variable_ident.field.is_some() {
            let mut declaration_ident = variable_ident.clone();
            declaration_ident.field = None;
            let declaration = self
                .variable_uses
                .get_mut(&declaration_ident)
                .unwrap_or_else(|| panic!("Struct {:?} does not exist", declaration_ident));
            declaration.num_used += 1;
        }
    }

    /// During codegen, tracks a variable declaration.
    /// This must be done again to know on what repeat a use occurs
    pub(crate) fn codegen_declare(&mut self, name: String, scope: u8) {
        let key = VariableKey::new(name.clone(), scope);
        if let Some(count) = self.codegen_repeats.get_mut(&key) {
            *count += 1;
        } else {
            self.codegen_repeats.insert(key, 0);
        }
    }

    /// During codegen, tracks a variable use.
    pub(crate) fn codegen_reuse(
        &mut self,
        name: String,
        scope: u8,
        field: Option<String>,
    ) -> Result<(bool, bool), VariableReuseError> {
        let scopes_declared = self
            .scopes_declared
            .get(&name)
            .ok_or_else(|| VariableNotFound::new(name.clone(), scope, field.clone()))?;
        let scope_declared = *scopes_declared
            .iter()
            .filter(|s| **s <= scope)
            .max()
            .ok_or_else(|| VariableNotFound::new(name.clone(), scope, field.clone()))?;

        let key = VariableKey::new(name.clone(), scope_declared);
        let repeat = self.codegen_repeats.get(&key).unwrap_or(&0);
        let ident = VariableIdent::new(name.clone(), *repeat, scope_declared, field.clone());

        let should_clone_parent = if field.is_some() {
            let struct_ident = VariableIdent::new(name.clone(), *repeat, scope_declared, None);
            let parent_analysis = self
                .variable_uses
                .get_mut(&struct_ident)
                .ok_or_else(|| VariableNotFound::new(name.clone(), scope_declared, None))?;

            parent_analysis.num_used -= 1;
            parent_analysis.should_clone()
        } else {
            false
        };

        let analysis = self
            .variable_uses
            .get_mut(&ident)
            .ok_or_else(|| VariableNotFound::new(name, scope_declared, field))?;

        analysis.num_used -= 1;
        let should_clone =
            analysis.should_clone() || should_clone_parent || scope_declared != scope;
        Ok((should_clone, analysis.is_comptime))
    }

    pub fn set_as_comptime(
        &mut self,
        name: String,
        scope: u8,
        field: Option<String>,
    ) -> Result<(), VariableReuseError> {
        let scopes_declared = self
            .scopes_declared
            .get(&name)
            .ok_or_else(|| VariableNotFound::new(name.clone(), scope, field.clone()))?;
        let scope_declared = *scopes_declared
            .iter()
            .filter(|s| **s <= scope)
            .max()
            .ok_or_else(|| VariableNotFound::new(name.clone(), scope, field.clone()))?;

        let key = VariableKey::new(name.clone(), scope_declared);
        let repeat = self.codegen_repeats.get(&key).unwrap_or(&0);
        let ident = VariableIdent::new(name.clone(), *repeat, scope_declared, field.clone());

        let analysis = self
            .variable_uses
            .get_mut(&ident)
            .ok_or_else(|| VariableNotFound::new(name, scope_declared, field))?;

        analysis.is_comptime = true;

        Ok(())
    }
}

#[derive(new, Debug)]
pub struct VariableNotFound {
    _name: String,
    _scope: u8,
    _field: Option<String>,
}

#[derive(Debug)]
#[allow(dead_code)]
pub enum VariableReuseError {
    VariableNotFound(VariableNotFound),
}

impl From<VariableNotFound> for VariableReuseError {
    fn from(value: VariableNotFound) -> Self {
        Self::VariableNotFound(value)
    }
}
