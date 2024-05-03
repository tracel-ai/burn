use std::collections::{HashMap, HashSet};

use syn::Stmt;

use crate::VariableKey;

#[derive(Debug)]
pub(crate) struct VariableAnalysis {
    num_used: usize,
    loop_level_declared: usize,
}

impl VariableAnalysis {
    pub fn should_clone(&mut self, loop_level: usize) -> bool {
        if self.num_used > 1 {
            self.num_used -= 1;
            true
        } else {
            self.loop_level_declared < loop_level
        }
    }
}

#[derive(Debug)]
pub(crate) struct CodeAnalysis {
    pub needed_functions: HashSet<VariableKey>,
    pub variable_analyses: HashMap<VariableKey, VariableAnalysis>,
}

#[derive(Debug, Default)]
pub(crate) struct CodeAnalysisBuilder {
    declarations: Vec<(VariableKey, usize)>,
    var_uses: Vec<VariableKey>,
    function_calls: HashSet<VariableKey>,
}

impl CodeAnalysis {
    pub fn should_clone(&mut self, ident: &syn::Ident, loop_level: usize) -> bool {
        let key: VariableKey = ident.into();
        match self.variable_analyses.remove(&key) {
            Some(mut var) => {
                let should_clone = var.should_clone(loop_level);
                self.variable_analyses.insert(key, var);
                return should_clone;
            }
            None => panic!("Ident {ident} not part of analysis"),
        }
    }

    pub fn create(func: &syn::ItemFn) -> CodeAnalysis {
        let code_analysis_builder = CodeAnalysisBuilder::default();
        code_analysis_builder.analyze(func)
    }
}

impl CodeAnalysisBuilder {
    fn analyze(mut self, func: &syn::ItemFn) -> CodeAnalysis {
        // Build the vector of (Id, depth), using recursion
        self.signature_declarations(&func.sig);
        self.stmts_occurrences(&func.block.stmts, 0);

        CodeAnalysis {
            variable_analyses: self.to_map(),
            needed_functions: self.function_calls,
        }
    }

    fn to_map(&self) -> HashMap<VariableKey, VariableAnalysis> {
        // Run through the vec and build hashmap, without recursion
        let mut variable_analyses = HashMap::<VariableKey, VariableAnalysis>::new();
        for declaration in self.declarations.iter() {
            let id = declaration.0.clone();
            let new_analysis = match variable_analyses.remove(&id) {
                Some(_) => {
                    panic!("Multiple variables with the same identifier is not supported")
                }
                None => VariableAnalysis {
                    num_used: 0,
                    loop_level_declared: declaration.1,
                },
            };

            variable_analyses.insert(id, new_analysis);
        }

        for id in self.var_uses.iter() {
            let prev_analysis = variable_analyses.remove(&id).expect(&format!(
                "Variable {:?} should be declared before it's used",
                id
            ));
            let new_analysis = VariableAnalysis {
                num_used: prev_analysis.num_used + 1,
                loop_level_declared: prev_analysis.loop_level_declared,
            };
            variable_analyses.insert(id.clone(), new_analysis);
        }

        variable_analyses
    }

    fn signature_declarations(&mut self, sig: &syn::Signature) {
        for input in &sig.inputs {
            match input {
                syn::FnArg::Typed(pat) => {
                    let ident = &*pat.pat;
                    match ident {
                        syn::Pat::Ident(pat_ident) => {
                            let id = &pat_ident.ident;
                            self.declarations.push((id.into(), 0));
                        }
                        _ => todo!("Analysis: unsupported ident {ident:?}"),
                    }
                }
                _ => todo!("Analysis: unsupported input {input:?}"),
            }
        }
    }

    fn stmts_occurrences(&mut self, stmts: &Vec<Stmt>, depth: usize) {
        for stmt in stmts {
            match stmt {
                // Declaration
                syn::Stmt::Local(local) => {
                    let id = match &local.pat {
                        syn::Pat::Ident(pat_ident) => Some(&pat_ident.ident),
                        syn::Pat::Type(pat_type) => Some(match &*pat_type.pat {
                            syn::Pat::Ident(pat_ident) => &pat_ident.ident,
                            _ => todo!("Analysis: unsupported typed path {:?}", pat_type.pat),
                        }),
                        syn::Pat::Wild(_) => None,
                        _ => todo!("Analysis: unsupported path {:?}", local.pat),
                    };
                    if let Some(id) = id {
                        self.declarations.push((id.into(), depth));
                    }
                    if let Some(local_init) = &local.init {
                        self.expr_occurrences(&local_init.expr, depth)
                    }
                }
                syn::Stmt::Expr(expr, _) => self.expr_occurrences(expr, depth),
                _ => todo!("Analysis: unsupported stmt {stmt:?}"),
            }
        }
    }

    fn expr_occurrences(&mut self, expr: &syn::Expr, depth: usize) {
        match expr {
            syn::Expr::ForLoop(expr) => {
                let depth = depth + 1;

                // Declaration of iterator
                if let syn::Pat::Ident(pat_ident) = &*expr.pat {
                    let id = &pat_ident.ident;
                    self.declarations.push((id.into(), depth));
                }

                self.stmts_occurrences(&expr.body.stmts, depth);
            }
            syn::Expr::While(expr) => {
                let depth = depth + 1;

                self.expr_occurrences(&expr.cond, depth);
                self.stmts_occurrences(&expr.body.stmts, depth);
            }
            syn::Expr::If(expr) => {
                if expr.else_branch.is_some() {
                    todo!("Analysis: else branch not supported");
                }

                self.expr_occurrences(&expr.cond, depth);
                self.stmts_occurrences(&expr.then_branch.stmts, depth);
            }
            syn::Expr::Assign(expr) => {
                self.expr_occurrences(&expr.left, depth);
                self.expr_occurrences(&expr.right, depth);
            }
            syn::Expr::Index(expr) => {
                self.expr_occurrences(&expr.expr, depth);
                self.expr_occurrences(&expr.index, depth);
            }
            syn::Expr::Path(expr) => {
                let ident = expr
                    .path
                    .get_ident()
                    .expect("Analysis: only ident path are supported.");

                // Use
                self.var_uses.push(ident.into());
            }
            syn::Expr::Binary(expr) => {
                self.expr_occurrences(&expr.left, depth);
                self.expr_occurrences(&expr.right, depth);
            }
            syn::Expr::Lit(_) => {}
            syn::Expr::Call(expr) => {
                match &*expr.func {
                    syn::Expr::Path(expr_path) => {
                        let ident = expr_path
                            .path
                            .get_ident()
                            .expect("Analysis: only ident supported for function call");
                        self.function_calls.insert(ident.into());
                    }
                    _ => todo!("Analysis: unsupported func expr {:?}", expr.func),
                }
                for arg in expr.args.iter() {
                    self.expr_occurrences(arg, depth);
                }
            }
            syn::Expr::MethodCall(expr) => {
                self.expr_occurrences(&expr.receiver, depth);
                for arg in expr.args.iter() {
                    self.expr_occurrences(arg, depth);
                }
            }
            _ => todo!("Analysis: unsupported expr {expr:?}"),
        }
    }
}
