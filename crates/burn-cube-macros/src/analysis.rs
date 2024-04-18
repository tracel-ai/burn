use std::collections::HashMap;

use syn::Stmt;

use crate::VariableKey;

#[derive(Debug)]
pub(crate) struct VariableAnalysis {
    num_used: usize,
    loop_level_declared: usize,
}

impl VariableAnalysis {
    pub fn should_clone(&self, loop_level: usize) -> bool {
        self.num_used > 1 || self.loop_level_declared < loop_level
    }
}

#[derive(Debug)]
pub(crate) struct VariableAnalyses {
    pub analyses: HashMap<VariableKey, VariableAnalysis>,
}

impl VariableAnalyses {
    pub fn should_clone(&self, ident: &syn::Ident, loop_level: usize) -> bool {
        let key: VariableKey = ident.into();
        if let Some(var) = self.analyses.get(&key) {
            return var.should_clone(loop_level);
        }

        false
    }

    pub fn create(func: &syn::ItemFn) -> Self {
        let analyses = analyze(func);

        Self { analyses }
    }
}

pub(crate) fn analyze(func: &syn::ItemFn) -> HashMap<VariableKey, VariableAnalysis> {
    // Build the vector of (Id, depth), using recursion
    let mut declarations = Vec::new();
    let mut var_uses = Vec::new();
    list_occurrences(&func.block.stmts, 0, &mut declarations, &mut var_uses);

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
        let prev_analysis = analyses
            .remove(&id)
            .expect("Variable {id} should be declared before it's used");
        let new_analysis = VariableAnalysis {
            num_used: prev_analysis.num_used + 1,
            loop_level_declared: prev_analysis.loop_level_declared,
        };
        analyses.insert(id, new_analysis);
    }

    analyses
}

fn list_occurrences(
    stmts: &Vec<Stmt>,
    depth: usize,
    declarations: &mut Vec<(VariableKey, usize)>,
    uses: &mut Vec<VariableKey>,
) {
    for stmt in stmts {
        match stmt {
            // Declaration
            syn::Stmt::Local(local) => match &local.pat {
                syn::Pat::Ident(pat_ident) => {
                    let id = &pat_ident.ident;
                    declarations.push((id.into(), depth));
                }
                _ => todo!(),
            },
            syn::Stmt::Expr(expr, _) => occ_expr(expr, depth, declarations, uses),
            _ => todo!(),
        }
    }
}

fn occ_expr(
    expr: &syn::Expr,
    depth: usize,
    declarations: &mut Vec<(VariableKey, usize)>,
    uses: &mut Vec<VariableKey>,
) {
    match expr {
        syn::Expr::ForLoop(expr) => {
            // Declaration
            if let syn::Pat::Ident(pat_ident) = &*expr.pat {
                let id = &pat_ident.ident;
                declarations.push((id.into(), depth));
            }

            list_occurrences(&expr.body.stmts, depth + 1, declarations, uses);
        }
        syn::Expr::Assign(expr) => {
            occ_expr(&expr.right, depth, declarations, uses);
        }
        syn::Expr::Index(expr) => panic!("{expr:?}"),
        syn::Expr::Path(expr) => panic!("{expr:?}"),
        syn::Expr::Binary(expr) => {
            
        },
        _ => todo!(),
    }
}
