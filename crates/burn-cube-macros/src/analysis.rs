use std::collections::HashMap;

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
pub(crate) struct VariableAnalyses {
    pub analyses: HashMap<VariableKey, VariableAnalysis>,
}

impl VariableAnalyses {
    pub fn should_clone(&mut self, ident: &syn::Ident, loop_level: usize) -> bool {
        let key: VariableKey = ident.into();
        match self.analyses.remove(&key) {
            Some(mut var) => {
                let should_clone = var.should_clone(loop_level);
                self.analyses.insert(key, var);
                return should_clone;
            }
            None => panic!("Ident {ident} not part of analysis"),
        }
    }

    pub fn create(func: &syn::ItemFn) -> Self {
        let analyses = analyze(func);

        Self { analyses }
    }
}

pub(crate) fn analyze(func: &syn::ItemFn) -> HashMap<VariableKey, VariableAnalysis> {
    // Build the vector of (Id, depth), using recursion
    let mut declarations = Vec::new();
    signature_declarations(&func.sig, &mut declarations);

    let mut var_uses = Vec::new();
    stmts_occurrences(&func.block.stmts, 0, &mut declarations, &mut var_uses);

    // Run through the vec and build hashmap, without recursion
    let mut analyses = HashMap::<VariableKey, VariableAnalysis>::new();
    for declaration in declarations.into_iter() {
        let id = declaration.0;
        let new_analysis = match analyses.remove(&id) {
            Some(_) => {
                panic!("Multiple variables with the same identifier is not supported")
            }
            None => VariableAnalysis {
                num_used: 0,
                loop_level_declared: declaration.1,
            },
        };

        analyses.insert(id, new_analysis);
    }

    for id in var_uses.into_iter() {
        let prev_analysis = analyses.remove(&id).expect(&format!(
            "Variable {:?} should be declared before it's used",
            id
        ));
        let new_analysis = VariableAnalysis {
            num_used: prev_analysis.num_used + 1,
            loop_level_declared: prev_analysis.loop_level_declared,
        };
        analyses.insert(id, new_analysis);
    }

    analyses
}

fn signature_declarations(sig: &syn::Signature, declarations: &mut Vec<(VariableKey, usize)>) {
    for input in &sig.inputs {
        match input {
            syn::FnArg::Typed(pat) => {
                let ident = &*pat.pat;
                match ident {
                    syn::Pat::Ident(pat_ident) => {
                        let id = &pat_ident.ident;
                        declarations.push((id.into(), 0));
                    }
                    _ => todo!("Analysis: unsupported ident {ident:?}"),
                }
            }
            _ => todo!("Analysis: unsupported input {input:?}"),
        }
    }
}

fn stmts_occurrences(
    stmts: &Vec<Stmt>,
    depth: usize,
    declarations: &mut Vec<(VariableKey, usize)>,
    uses: &mut Vec<VariableKey>,
) {
    for stmt in stmts {
        match stmt {
            // Declaration
            syn::Stmt::Local(local) => {
                let id = match &local.pat {
                    syn::Pat::Ident(pat_ident) => &pat_ident.ident,
                    syn::Pat::Type(pat_type) => match &*pat_type.pat {
                        syn::Pat::Ident(pat_ident) => &pat_ident.ident,
                        _ => todo!("Analysis: unsupported typed path {:?}", pat_type.pat),
                    },
                    _ => todo!("Analysis: unsupported path {:?}", local.pat),
                };
                declarations.push((id.into(), depth));
                if let Some(local_init) = &local.init {
                    expr_occurrences(&local_init.expr, depth, declarations, uses)
                }
            }
            syn::Stmt::Expr(expr, _) => expr_occurrences(expr, depth, declarations, uses),
            _ => todo!("Analysis: unsupported stmt {stmt:?}"),
        }
    }
}

fn expr_occurrences(
    expr: &syn::Expr,
    depth: usize,
    declarations: &mut Vec<(VariableKey, usize)>,
    uses: &mut Vec<VariableKey>,
) {
    match expr {
        syn::Expr::ForLoop(expr) => {
            let depth = depth + 1;

            // Declaration of iterator
            if let syn::Pat::Ident(pat_ident) = &*expr.pat {
                let id = &pat_ident.ident;
                declarations.push((id.into(), depth));
            }

            stmts_occurrences(&expr.body.stmts, depth, declarations, uses);
        }
        syn::Expr::While(expr) => {
            let depth = depth + 1;

            expr_occurrences(&expr.cond, depth, declarations, uses);
            stmts_occurrences(&expr.body.stmts, depth, declarations, uses);
        }
        syn::Expr::Assign(expr) => {
            expr_occurrences(&expr.left, depth, declarations, uses);
            expr_occurrences(&expr.right, depth, declarations, uses);
        }
        syn::Expr::Index(expr) => {
            expr_occurrences(&expr.expr, depth, declarations, uses);
            expr_occurrences(&expr.index, depth, declarations, uses);
        }
        syn::Expr::Path(expr) => {
            let ident = expr
                .path
                .get_ident()
                .expect("Analysis: only ident path are supported.");

            // Use
            uses.push(ident.into());
        }
        syn::Expr::Binary(expr) => {
            expr_occurrences(&expr.left, depth, declarations, uses);
            expr_occurrences(&expr.right, depth, declarations, uses);
        }
        syn::Expr::MethodCall(expr) => {
            if expr.args.is_empty() {
                panic!("Analysis: method call with args is unsupported")
            }
        }
        _ => todo!("Analysis: unsupported expr {expr:?}"),
    }
}
