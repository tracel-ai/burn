use syn::{Member, Pat, PathArguments, Stmt};

use crate::tracker::VariableTracker;

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

#[derive(Debug, Default)]
/// Reads the whole Cube code and accumulates information,
/// to generate a VariableTracker that looked variable uses ahead
pub(crate) struct VariableAnalyzer {
    variable_tracker: VariableTracker,
}

impl VariableAnalyzer {
    pub fn create_tracker(func: &syn::ItemFn) -> VariableTracker {
        let analyzer = VariableAnalyzer::default();
        analyzer.analyze(func)
    }
}

impl VariableAnalyzer {
    fn analyze(mut self, func: &syn::ItemFn) -> VariableTracker {
        // Build the vector of (Id, depth), using recursion
        self.signature_declarations(&func.sig);
        self.find_occurrences_in_stmts(&func.block.stmts, 0);

        self.variable_tracker
    }

    fn signature_declarations(&mut self, sig: &syn::Signature) {
        for input in &sig.inputs {
            match input {
                syn::FnArg::Typed(pat) => {
                    let ident = &*pat.pat;
                    let is_comptime = is_ty_comptime(&pat.ty);

                    match ident {
                        syn::Pat::Ident(pat_ident) => {
                            let id = &pat_ident.ident;
                            self.variable_tracker
                                .analyze_declare(id.to_string(), 0, is_comptime);
                        }
                        _ => todo!("Analysis: unsupported ident {ident:?}"),
                    }
                }
                _ => todo!("Analysis: unsupported input {input:?}"),
            }
        }
    }

    fn find_occurrences_in_stmts(&mut self, stmts: &Vec<Stmt>, depth: u8) {
        for stmt in stmts {
            match stmt {
                // Declaration
                syn::Stmt::Local(local) => {
                    let mut is_comptime = false;
                    let id = match &local.pat {
                        syn::Pat::Ident(pat_ident) => Some(&pat_ident.ident),
                        syn::Pat::Type(pat_type) => {
                            is_comptime = is_ty_comptime(&pat_type.ty);
                            match &*pat_type.pat {
                                syn::Pat::Ident(pat_ident) => Some(&pat_ident.ident),
                                _ => todo!("Analysis: unsupported typed path {:?}", pat_type.pat),
                            }
                        }
                        syn::Pat::Wild(_) => None,
                        _ => todo!("Analysis: unsupported path {:?}", local.pat),
                    };
                    if let Some(id) = id {
                        self.variable_tracker
                            .analyze_declare(id.to_string(), depth, is_comptime);
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

    fn find_occurrences_in_expr(&mut self, expr: &syn::Expr, depth: u8) {
        match expr {
            syn::Expr::ForLoop(expr) => {
                self.find_occurrences_in_expr(&expr.expr, depth);

                let depth = depth + 1;

                if let syn::Pat::Ident(pat_ident) = &*expr.pat {
                    let id = &pat_ident.ident;
                    self.variable_tracker
                        .analyze_declare(id.to_string(), depth, false);
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
                        // Unsupported: handled in codegen.
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
                    self.variable_tracker.analyze_reuse(ident, depth, None);
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
                    // Unsupported: handled in codegen.
                }
            }
            syn::Expr::Paren(expr) => self.find_occurrences_in_expr(&expr.expr, depth),
            syn::Expr::Array(_expr) => {
                // No analysis since only literals are supported
            }
            syn::Expr::Reference(expr) => self.find_occurrences_in_expr(&expr.expr, depth),
            syn::Expr::Closure(expr) => {
                let depth = depth + 1;

                for path in expr.inputs.iter() {
                    let mut is_comptime = false;
                    let ident = match path {
                        Pat::Ident(pat_ident) => &pat_ident.ident,
                        Pat::Type(pat_type) => {
                            is_comptime = is_ty_comptime(&pat_type.ty);

                            if let Pat::Ident(pat_ident) = &*pat_type.pat {
                                &pat_ident.ident
                            } else {
                                todo!("Analysis: {:?} not supported in closure inputs. ", path);
                            }
                        }
                        _ => todo!("Analysis: {:?} not supported in closure inputs. ", path),
                    };

                    self.variable_tracker
                        .analyze_declare(ident.to_string(), depth, is_comptime);
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

                        self.variable_tracker.analyze_reuse(
                            struct_ident,
                            depth,
                            Some(attribute_ident.to_string()),
                        );
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
            syn::Expr::Range(_range) => {
                // Error is handled during codegen.
            }
            _ => {
                // Error is handled during codegen.
            }
        }
    }
}

fn is_ty_comptime(ty: &syn::Type) -> bool {
    if let syn::Type::Path(path) = ty {
        for segment in path.path.segments.iter() {
            if segment.ident == "Comptime" {
                return true;
            }
        }
    }

    false
}
