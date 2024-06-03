use std::collections::HashMap;

use syn::{Member, Pat, PathArguments, Stmt};

use crate::variable_key::VariableKey;

pub const KEYWORDS: [&str; 20] = [
    "ABSOLUTE_POS",
    "ABSOLUTE_POS_X",
    "ABSOLUTE_POS_Y",
    "ABSOLUTE_POS_Z",
    "UNIT_POS",
    "UNIT_POS_X",
    "UNIT_POS_Y",
    "UNIT_POS_Z",
    "CUBE_POS",
    "CUBE_POS_X",
    "CUBE_POS_Y",
    "CUBE_POS_Z",
    "CUBE_DIM",
    "CUBE_DIM_X",
    "CUBE_DIM_Y",
    "CUBE_DIM_Z",
    "CUBE_COUNT",
    "CUBE_COUNT_X",
    "CUBE_COUNT_Y",
    "CUBE_COUNT_Z",
];

#[derive(Debug)]
/// Information about a single variable's use in Cube code
/// Information about a single variable's use in Cube code
/// Useful to figure out when the generated variable will need cloning
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
/// Information about all variables in the Cube code, transmitted to codegen
pub(crate) struct CodeAnalysis {
    pub variable_analyses: HashMap<VariableKey, VariableAnalysis>,
}

#[derive(Debug, Default)]
/// Reads the Cube code and accumulates information, to generate a CodeAnalysis artefact
pub(crate) struct CodeAnalysisBuilder {
    declarations: Vec<(VariableKey, usize)>,
    var_uses: Vec<VariableKey>,
}

impl CodeAnalysis {
    pub fn should_clone(&mut self, key: VariableKey, loop_level: usize) -> bool {
        match self.variable_analyses.remove(&key) {
            Some(mut var) => {
                let should_clone = var.should_clone(loop_level);
                self.variable_analyses.insert(key.clone(), var);

                should_clone
                    || match key {
                        VariableKey::LocalKey(_) => false,
                        VariableKey::Attribute((struct_, _)) => {
                            self.variable_analyses
                                .get(&struct_.into())
                                .unwrap()
                                .num_used
                                > 0
                        }
                    }
            }
            None => panic!("Variable {key:?} not part of analysis"),
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
        self.find_occurrences_in_stmts(&func.block.stmts, 0);

        CodeAnalysis {
            variable_analyses: self.to_map(),
        }
    }

    fn to_map(&self) -> HashMap<VariableKey, VariableAnalysis> {
        // Run through the vec and build hashmap, without recursion
        let mut variable_analyses = HashMap::<VariableKey, VariableAnalysis>::new();
        for declaration in self.declarations.iter() {
            let id = declaration.0.clone();
            let new_analysis = match variable_analyses.remove(&id) {
                Some(_) => {
                    panic!("Analysis: Multiple variables with the same identifier is not supported")
                }
                None => VariableAnalysis {
                    num_used: 0,
                    loop_level_declared: declaration.1,
                },
            };

            variable_analyses.insert(id, new_analysis);
        }

        for id in self.var_uses.iter() {
            match variable_analyses.remove(id) {
                Some(prev_analysis) => {
                    let new_analysis = VariableAnalysis {
                        num_used: prev_analysis.num_used + 1,
                        loop_level_declared: prev_analysis.loop_level_declared,
                    };

                    variable_analyses.insert(id.clone(), new_analysis);
                }
                None => {
                    if let VariableKey::Attribute((struct_, _)) = id {
                        let struct_analysis = variable_analyses.get(&((struct_.clone()).into())).unwrap_or_else(||
                            panic!(
                                "Analysis: Struct {:?} should be declared before using its attribute",
                                struct_
                            )
                        );
                        let attr_analysis = VariableAnalysis {
                            num_used: 1,
                            loop_level_declared: struct_analysis.loop_level_declared,
                        };
                        variable_analyses.insert(id.clone(), attr_analysis);
                    } else {
                        panic!(
                            "Analysis: Variable {:?} should be declared before it's used",
                            id
                        )
                    }
                }
            };
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

    fn find_occurrences_in_stmts(&mut self, stmts: &Vec<Stmt>, depth: usize) {
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
                        self.find_occurrences_in_expr(&local_init.expr, depth)
                    }
                }
                syn::Stmt::Expr(expr, _) => self.find_occurrences_in_expr(expr, depth),
                _ => todo!("Analysis: unsupported stmt {stmt:?}"),
            }
        }
    }

    fn find_occurrences_in_expr(&mut self, expr: &syn::Expr, depth: usize) {
        match expr {
            syn::Expr::ForLoop(expr) => {
                self.find_occurrences_in_expr(&expr.expr, depth);

                let depth = depth + 1;

                // Declaration of iterator
                if let syn::Pat::Ident(pat_ident) = &*expr.pat {
                    let id = &pat_ident.ident;
                    self.declarations.push((id.into(), depth));
                }

                self.find_occurrences_in_stmts(&expr.body.stmts, depth);
            }
            syn::Expr::While(expr) => {
                let depth = depth + 1;

                self.find_occurrences_in_expr(&expr.cond, depth);
                self.find_occurrences_in_stmts(&expr.body.stmts, depth);
            }
            syn::Expr::Loop(expr) => {
                let depth = depth + 1;

                self.find_occurrences_in_stmts(&expr.body.stmts, depth);
            }
            syn::Expr::If(expr) => {
                let depth = depth + 1;

                self.find_occurrences_in_expr(&expr.cond, depth);
                self.find_occurrences_in_stmts(&expr.then_branch.stmts, depth);
                if let Some((_, expr)) = &expr.else_branch {
                    if let syn::Expr::Block(expr_block) = &**expr {
                        self.find_occurrences_in_stmts(&expr_block.block.stmts, depth);
                    } else {
                        todo!("Analysis: Only block else expr is supported")
                    }
                }
            }
            syn::Expr::Assign(expr) => {
                self.find_occurrences_in_expr(&expr.left, depth);
                self.find_occurrences_in_expr(&expr.right, depth);
            }
            syn::Expr::Index(expr) => {
                self.find_occurrences_in_expr(&expr.expr, depth);
                self.find_occurrences_in_expr(&expr.index, depth);
            }
            syn::Expr::Path(expr) => {
                let ident = expr
                    .path
                    .get_ident()
                    .expect("Analysis: only ident path are supported.");

                if !KEYWORDS.contains(&ident.to_string().as_str()) {
                    // Use
                    self.var_uses.push(ident.into());
                }
            }
            syn::Expr::Binary(expr) => {
                self.find_occurrences_in_expr(&expr.left, depth);
                self.find_occurrences_in_expr(&expr.right, depth);
            }
            syn::Expr::Lit(_) => {}
            syn::Expr::Call(expr) => {
                match &*expr.func {
                    syn::Expr::Path(expr_path) => {
                        if let Some(first_segment) = expr_path.path.segments.first() {
                            // Check if the path segment has generic arguments
                            if let PathArguments::AngleBracketed(arguments) =
                                &first_segment.arguments
                            {
                                // Extract the generic arguments
                                for arg in &arguments.args {
                                    match arg {
                                        syn::GenericArgument::Type(_)
                                        | syn::GenericArgument::Constraint(_) => {}
                                        _ => todo!("Analysis: Generic {:?} not supported", arg),
                                    }
                                }
                            }
                        }
                    }
                    _ => todo!("Analysis: unsupported func expr {:?}", expr.func),
                }
                for arg in expr.args.iter() {
                    self.find_occurrences_in_expr(arg, depth);
                }
            }
            syn::Expr::MethodCall(expr) => {
                self.find_occurrences_in_expr(&expr.receiver, depth);
                for arg in expr.args.iter() {
                    self.find_occurrences_in_expr(arg, depth);
                }
            }
            syn::Expr::Break(_) => {}
            syn::Expr::Return(expr) => {
                if expr.expr.is_some() {
                    todo!("Analysis: only void return supported")
                }
            }
            syn::Expr::Paren(expr) => self.find_occurrences_in_expr(&expr.expr, depth),
            syn::Expr::Array(expr) => {
                for element in expr.elems.iter() {
                    match element {
                        syn::Expr::Lit(_) => {}
                        _ => todo!("Analysis: only array of literals is supported"),
                    }
                }
            }
            syn::Expr::Reference(expr) => self.find_occurrences_in_expr(&expr.expr, depth),
            syn::Expr::Closure(expr) => {
                let depth = depth + 1;

                for path in expr.inputs.iter() {
                    let ident = match path {
                        Pat::Ident(pat_ident) => &pat_ident.ident,
                        Pat::Type(pat_type) => {
                            if let Pat::Ident(pat_ident) = &*pat_type.pat {
                                &pat_ident.ident
                            } else {
                                todo!("Analysis: {:?} not supported in closure inputs. ", path);
                            }
                        }
                        _ => todo!("Analysis: {:?} not supported in closure inputs. ", path),
                    };

                    self.declarations.push(((ident).into(), depth));
                }

                self.find_occurrences_in_expr(&expr.body, depth)
            }
            syn::Expr::Unary(expr) => self.find_occurrences_in_expr(&expr.expr, depth),
            syn::Expr::Field(expr) => {
                if let Member::Named(attribute_ident) = &expr.member {
                    if let syn::Expr::Path(struct_expr) = &*expr.base {
                        let struct_ident = struct_expr
                            .path
                            .get_ident()
                            .expect("Analysis: field access only supported on ident struct.");

                        self.var_uses.push((struct_ident, attribute_ident).into())
                    } else {
                        todo!("Analysis: field access only supported on ident struct.");
                    }
                } else {
                    todo!("Analysis: unnamed attribute not supported.");
                }
            }
            syn::Expr::Struct(expr) => {
                for field in expr.fields.iter() {
                    self.find_occurrences_in_expr(&field.expr, depth)
                }
            }
            _ => todo!("Analysis: unsupported expr {expr:?}"),
        }
    }
}
