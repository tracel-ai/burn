use std::collections::HashMap;

#[derive(new, Hash, PartialEq, Eq, Debug, Clone)]
pub struct VariableIdent {
    pub name: String,
    repeat: u8,
    scope: u8,
    pub field: Option<String>,
}

#[derive(new, Eq, PartialEq, Hash, Debug)]
struct VariableKey {
    name: String,
    scope: u8,
}

#[derive(Debug, Default)]
pub(crate) struct VariableReuseAnalyzer {
    analysis_repeats: HashMap<VariableKey, u8>,
    scopes_declared: HashMap<String, Vec<u8>>,
    codegen_repeats: HashMap<VariableKey, u8>,
    analyses: HashMap<VariableIdent, VariableAnalysis>,
}

#[derive(Debug, Default)]
pub(crate) struct VariableAnalysis {
    pub num_used: usize,
}

impl VariableAnalysis {
    pub fn should_clone(&self) -> bool {
        self.num_used > 1
    }
}

impl VariableReuseAnalyzer {
    pub(crate) fn analyze_declare(&mut self, name: String, scope: u8) {
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

        let analysis = VariableAnalysis { num_used: 1 };
        let variable_ident = VariableIdent::new(name, repeat, scope, None);
        self.analyses.insert(variable_ident, analysis);
    }

    pub(crate) fn analyze_reuse(&mut self, name: String, scope: u8, field: Option<String>) {
        let scopes_declared = self
            .scopes_declared
            .get(&name)
            .expect("Analysis: Variable should be declared.");
        let scope = *scopes_declared
            .iter()
            .filter(|s| **s <= scope)
            .max()
            .unwrap();
        let key = VariableKey::new(name.clone(), scope);
        if let Some(repeat) = self.analysis_repeats.get(&key) {
            let variable = VariableIdent::new(name, *repeat, scope, field);
            self.analyze(&variable);
        }
    }

    fn analyze(&mut self, variable_ident: &VariableIdent) {
        match self.analyses.get_mut(variable_ident) {
            Some(analysis) => {
                analysis.num_used += 1;

                if variable_ident.field.is_some() {
                    let mut declaration_ident = variable_ident.clone();
                    declaration_ident.field = None;
                    let declaration = self
                        .analyses
                        .get_mut(&declaration_ident)
                        .expect(&format!("Struct {:?} does not exist", declaration_ident));
                    declaration.num_used += 1;
                }

                return;
            }
            None => {
                if variable_ident.field.is_some() {
                    let mut declaration_ident = variable_ident.clone();
                    declaration_ident.field = None;
                    let declaration = self
                        .analyses
                        .get_mut(&declaration_ident)
                        .expect(&format!("Struct {:?} does not exist", declaration_ident));
                    declaration.num_used += 1;

                    let attr_analysis = VariableAnalysis { num_used: 1 };
                    self.analyses.insert(variable_ident.clone(), attr_analysis);
                    return;
                }
            }
        };

        panic!("Variable not declared")
    }

    pub(crate) fn codegen_declare(&mut self, name: String, scope: u8) {
        let key = VariableKey::new(name.clone(), scope);
        if let Some(count) = self.codegen_repeats.get_mut(&key) {
            *count += 1;
        } else {
            self.codegen_repeats.insert(key, 0);
        }
    }

    pub(crate) fn codegen_reuse(
        &mut self,
        name: String,
        scope: u8,
        field: Option<String>,
    ) -> Result<bool, VariableReuseError> {
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
                .analyses
                .get_mut(&struct_ident)
                .ok_or_else(|| VariableNotFound::new(name.clone(), scope_declared, None))?;

            parent_analysis.num_used -= 1;
            parent_analysis.should_clone()
        } else {
            false
        };

        let analysis = self
            .analyses
            .get_mut(&ident)
            .ok_or_else(|| VariableNotFound::new(name, scope_declared, field))?;

        analysis.num_used -= 1;
        Ok(analysis.should_clone() || should_clone_parent || scope_declared != scope)
    }
}

#[derive(new, Debug)]
pub struct VariableNotFound {
    _name: String,
    _scope: u8,
    _field: Option<String>,
}

#[derive(Debug)]
pub enum VariableReuseError {
    VariableNotFound(VariableNotFound),
}

impl From<VariableNotFound> for VariableReuseError {
    fn from(value: VariableNotFound) -> Self {
        Self::VariableNotFound(value)
    }
}
