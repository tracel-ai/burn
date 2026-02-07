use burn_core as burn;

use super::GradientsParams;
use crate::LearningRate;
use burn::config::Config;
use burn::module::{AutodiffModule, Module, ModuleMapper, ModuleVisitor, Param};
use burn::prelude::ToElement;
use burn::record::Record;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, backend::AutodiffBackend};
use serde::{Deserialize, Serialize};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float as _;

/// Cubic Interploate
///
/// Uses two points (x1, f1), (x2, f2) and their first derivatives g1,g2 to construct
/// a cubic interpolant and return its minimum within the given bounds.
fn cubic_interpolate(
    x1: f64,
    f1: f64,
    g1: f64,
    x2: f64,
    f2: f64,
    g2: f64,
    bounds: Option<(f64, f64)>,
) -> f64 {
    // Compute bounds of interpolation area
    let (min_bound, max_bound) = bounds.unwrap_or(if x1 <= x2 { (x1, x2) } else { (x2, x1) });
    // Code for most common case: cubic interpolation of 2 points
    // with function and derivative values for both
    // Soulution in this case (where x2 is the farthest point)
    // d1 = g1 + g2 - 3*(f1 - f2) / (x1-x2);
    // d2 = sqrt(d1^2 - g1 * g2);
    // min_pos = x2 - (x2 - x1)*((g2 + d2 - d1)/(g2 - g1 + 2*d2));
    // t_new = min(max(min_pos,min_bound), max_bound);
    let d1 = g1 + g2 - 3.0 * (f1 - f2) / (x1 - x2);
    let d2_square = d1 * d1 - g1 * g2;

    if d2_square >= 0.0 {
        let d2 = d2_square.sqrt();
        let min_pos = if x1 <= x2 {
            x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2.0 * d2))
        } else {
            x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2.0 * d2))
        };
        min_pos.max(min_bound).min(max_bound)
    } else {
        (min_bound + max_bound) / 2.0
    }
}
/// Auxiliray Struct For Strong_Wolfe
struct LineSearchSample<B: Backend> {
    // step size
    t: f64,
    // loss
    f: f64,
    // gradient
    g: Tensor<B, 1>,
    // directional derivative
    gtd: f64,
}

#[allow(clippy::too_many_arguments)]
fn strong_wolfe<B: Backend, F>(
    // obj_func(x,step size,direction) -> (loss,grad)
    obj_func: &mut F,
    x: &Tensor<B, 1>,
    // initial step size
    mut t: f64,
    d: &Tensor<B, 1>,
    f: f64,
    g: Tensor<B, 1>,
    gtd: f64,
    c1: f64,
    c2: f64,
    tolerance_change: f64,
    max_ls: usize,
) -> (f64, Tensor<B, 1>, f64, usize)
where
    F: FnMut(&Tensor<B, 1>, f64, &Tensor<B, 1>) -> (f64, Tensor<B, 1>),
{
    let d_norm = d.clone().abs().max().into_scalar().to_f64();

    // evaluate objective and gradient using initial step
    let (mut f_new, mut g_new) = obj_func(x, t, d);
    let mut ls_func_evals = 1;
    let mut gtd_new = g_new.clone().dot(d.clone()).into_scalar().to_f64();

    // bracket an interval [t_prev,t] containing a point satisfying the Wolfe criteria
    let (mut t_prev, mut f_prev, mut g_prev, mut gtd_prev) = (0.0, f, g.clone(), gtd);
    let mut done = false;
    let mut ls_iter = 0;

    // the interval [low,high] using for Zoom phase
    let mut bracket: Option<[LineSearchSample<B>; 2]> = None;
    // point which satisfy the wolfe condition
    let mut wolfe_bracket: Option<LineSearchSample<B>> = None;
    while ls_iter < max_ls {
        // Checking Conditions.

        // Checking the Armijo Condition and function value increasing condition.
        // Armijo: f(x+t*d) <= f(x) + c_1 t gtd
        if f_new > (f + c1 * t * gtd) || (ls_iter > 1 && f_new >= f_prev) {
            bracket = Some([
                LineSearchSample {
                    t: t_prev,
                    f: f_prev,
                    g: g_prev,
                    gtd: gtd_prev,
                },
                LineSearchSample {
                    t,
                    f: f_new,
                    g: g_new.clone(),
                    gtd: gtd_new,
                },
            ]);
            break;
        }

        // Checking Strong Wolfe Condition
        // |gtd_new| <= -c_2 gtd
        if gtd_new.abs() <= -c2 * gtd {
            wolfe_bracket = Some(LineSearchSample {
                t,
                f: f_new,
                g: g_new.clone(),
                gtd: gtd_new,
            });
            done = true;
            break;
        }

        // gtd_new >=0 , there must be a local minimum in the inteval.
        if gtd_new >= 0.0 {
            bracket = Some([
                LineSearchSample {
                    t: t_prev,
                    f: f_prev,
                    g: g_prev,
                    gtd: gtd_prev,
                },
                LineSearchSample {
                    t,
                    f: f_new,
                    g: g_new.clone(),
                    gtd: gtd_new,
                },
            ]);
            break;
        }

        // interpolate
        let min_step = t + 0.01 * (t - t_prev);
        let max_step = t * 10.0;
        let t_next = cubic_interpolate(
            t_prev,
            f_prev,
            gtd_prev,
            t,
            f_new,
            gtd_new,
            Some((min_step, max_step)),
        );
        t_prev = t;
        f_prev = f_new;
        g_prev = g_new;
        gtd_prev = gtd_new;

        // next step
        t = t_next;
        (f_new, g_new) = obj_func(x, t, d);
        ls_func_evals += 1;
        gtd_new = g_new.clone().dot(d.clone()).into_scalar().to_f64();
        ls_iter += 1;
    }
    if let Some(sample) = wolfe_bracket {
        return (sample.f, sample.g, sample.t, ls_func_evals);
    }

    let mut bracket = bracket.unwrap_or_else(|| {
        [
            LineSearchSample {
                t: 0.0,
                f,
                g: g.clone(),
                gtd,
            },
            LineSearchSample {
                t,
                f: f_new,
                g: g_new.clone(),
                gtd: gtd_new,
            },
        ]
    });

    // zoom phase
    let mut insuf_progress = false;

    // find high and low points in bracket
    let (mut low_idx, mut high_idx) = if bracket[0].f <= bracket[1].f {
        (0, 1)
    } else {
        (1, 0)
    };

    while !done && ls_iter < max_ls {
        let diff = (bracket[1].t - bracket[0].t).abs();
        // line-search bracket is so small
        if diff * d_norm < tolerance_change {
            break;
        }

        // compute new trial value
        t = cubic_interpolate(
            bracket[0].t,
            bracket[0].f,
            bracket[0].gtd,
            bracket[1].t,
            bracket[1].f,
            bracket[1].gtd,
            None,
        );

        let b_min = bracket[0].t.min(bracket[1].t);
        let b_max = bracket[0].t.max(bracket[1].t);
        let eps = 0.1 * (b_max - b_min);

        if (b_max - t).min(t - b_min) < eps {
            // interpolation close to boundary
            if insuf_progress || t >= b_max || t <= b_min {
                t = if (t - b_max).abs() < (t - b_min).abs() {
                    b_max - eps
                } else {
                    b_min + eps
                };
                insuf_progress = false;
            } else {
                insuf_progress = true;
            }
        } else {
            insuf_progress = false;
        }

        // Evaluate new point
        (f_new, g_new) = obj_func(x, t, d);

        ls_func_evals += 1;
        gtd_new = g_new.clone().dot(d.clone()).into_scalar().to_f64();
        ls_iter += 1;

        let armijo_holds = f_new <= (f + c1 * t * gtd) && f_new < bracket[low_idx].f;

        if !armijo_holds {
            bracket[high_idx] = LineSearchSample {
                t,
                f: f_new,
                g: g_new,
                gtd: gtd_new,
            };
        } else {
            if gtd_new.abs() <= -c2 * gtd {
                return (f_new, g_new, t, ls_func_evals);
            }

            if gtd_new * (bracket[high_idx].t - bracket[low_idx].t) >= 0.0 {
                bracket[high_idx] = LineSearchSample {
                    t: bracket[low_idx].t,
                    f: bracket[low_idx].f,
                    g: bracket[low_idx].g.clone(),
                    gtd: bracket[low_idx].gtd,
                };
            }
            bracket[low_idx] = LineSearchSample {
                t,
                f: f_new,
                g: g_new,
                gtd: gtd_new,
            };
        }

        if bracket[0].f <= bracket[1].f {
            low_idx = 0;
            high_idx = 1;
        } else {
            low_idx = 1;
            high_idx = 0;
        }
    }
    // return stuff
    (
        bracket[low_idx].f,
        bracket[low_idx].g.clone(),
        bracket[low_idx].t,
        ls_func_evals,
    )
}

/// Strategy for the line search optimization phase
#[derive(Clone, Default, Debug, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LineSearchFn {
    /// No line search performed
    #[default]
    None,
    /// strong wolfe conditions
    ///
    /// See: <https://en.wikipedia.org/wiki/Wolfe_conditions>
    StrongWolfe,
}

/// LBFGS Configuration.
#[derive(Config, Debug)]
pub struct LBFGSConfig {
    /// maximal number of iterations per optimization step (default: 20)
    #[config(default = 20)]
    pub max_iter: usize,
    /// update history size (default: 100).
    #[config(default = 100)]
    pub history_size: usize,
    /// ermination tolerance on first order optimality (default: 1e-7).
    #[config(default = 1e-7)]
    pub tolerance_grad: f64,
    /// termination tolerance on function value/parameter changes (default: 1e-9).
    #[config(default = 1e-9)]
    pub tolerance_change: f64,
    /// maximal number of function evaluations per optimization step (default: max_iter * 1.25).
    #[config(default = "None")]
    pub max_eval: Option<usize>,
    ///  either ‘strong_wolfe’ or None (default: None).
    #[config(default = "LineSearchFn::None")]
    pub line_search_fn: LineSearchFn,
}

impl LBFGSConfig {
    /// Initialize AdamW optimizer
    ///
    /// # Returns
    ///
    /// Returns an optimizer that can be used to optimize a module
    pub fn init<B: AutodiffBackend>(&self) -> LBFGS<B> {
        // by default max_eval = max_iter * 5/4
        let max_eval = self.max_eval.unwrap_or(self.max_iter * 5 / 4);
        LBFGS {
            config: LBFGSConfig {
                max_iter: self.max_iter,
                history_size: self.history_size,
                tolerance_grad: self.tolerance_grad,
                tolerance_change: self.tolerance_change,
                max_eval: Some(max_eval),
                line_search_fn: self.line_search_fn,
            },
            state: Default::default(),
        }
    }
}

/// Collects gradients in module visit order.
struct FlattenGradsVisitorInner<'a, B: AutodiffBackend> {
    grads: &'a GradientsParams,
    tensors: &'a mut Vec<Tensor<B::InnerBackend, 1>>,
}

impl<B: AutodiffBackend> ModuleVisitor<B> for FlattenGradsVisitorInner<'_, B> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        if let Some(g) = self.grads.get::<B::InnerBackend, D>(param.id) {
            let numel = g.shape().num_elements();
            self.tensors.push(g.reshape([numel]));
        }
    }
}

/// Flatten params to inner backend 1D tensor.
fn flatten_params_inner<B: AutodiffBackend, M: Module<B>>(
    module: &M,
) -> Tensor<B::InnerBackend, 1> {
    let mut tensors = Vec::new();
    let mut visitor = FlattenParamsVisitorInner::<B> {
        tensors: &mut tensors,
    };
    module.visit(&mut visitor);
    if tensors.is_empty() {
        return Tensor::empty([0], &module.devices()[0]);
    }
    Tensor::cat(tensors, 0)
}

struct FlattenParamsVisitorInner<'a, B: AutodiffBackend> {
    tensors: &'a mut Vec<Tensor<B::InnerBackend, 1>>,
}

impl<B: AutodiffBackend> ModuleVisitor<B> for FlattenParamsVisitorInner<'_, B> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        let t = param.val().inner();
        let numel = t.shape().num_elements();
        self.tensors.push(t.reshape([numel]));
    }
}

/// Flatten gradients for a module.
fn flatten_grads_inner<B: AutodiffBackend, M: Module<B>>(
    module: &M,
    grads: &GradientsParams,
) -> Tensor<B::InnerBackend, 1> {
    let mut tensors = Vec::new();
    let mut visitor = FlattenGradsVisitorInner {
        grads,
        tensors: &mut tensors,
    };
    module.visit(&mut visitor);
    if tensors.is_empty() {
        return Tensor::empty([0], &module.devices()[0]);
    }
    Tensor::cat(tensors, 0)
}

/// Mapper that assigns each float param from a flat inner-backend 1D tensor.
struct ParamsFromFlatMapperInner<'a, B: AutodiffBackend> {
    flat: &'a Tensor<B::InnerBackend, 1>,
    offset: &'a mut usize,
}

impl<B: AutodiffBackend> ParamsFromFlatMapperInner<'_, B> {
    fn take_slice(&mut self, numel: usize) -> Tensor<B::InnerBackend, 1> {
        let start = *self.offset;
        *self.offset += numel;
        self.flat.clone().slice(start..*self.offset)
    }
}

impl<B: AutodiffBackend> ModuleMapper<B> for ParamsFromFlatMapperInner<'_, B> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let (id, tensor, mapper) = param.consume();
        let numel = tensor.shape().num_elements();
        let slice_1d = self.take_slice(numel);
        let new_inner = slice_1d.reshape(tensor.shape());
        let new_tensor = Tensor::from_inner(new_inner).require_grad();
        Param::from_mapped_value(id, new_tensor, mapper)
    }
}

/// Overwrite module parameters from a flat inner-backend 1D tensor
fn set_params_from_flat_inner<B: AutodiffBackend, M: Module<B>>(
    module: M,
    flat: Tensor<B::InnerBackend, 1>,
) -> M {
    let mut offset = 0;
    let mut mapper = ParamsFromFlatMapperInner {
        flat: &flat,
        offset: &mut offset,
    };
    module.map(&mut mapper)
}

/// L-BFGS optimizer state
#[derive(Clone, Record)]
pub struct LBFGSState<B: Backend> {
    /// Historical displacement vectors
    pub history_s: Vec<Tensor<B, 1>>,
    /// Historical gradient difference vectors
    pub history_y: Vec<Tensor<B, 1>>,
}

impl<B: Backend> LBFGSState<B> {
    /// Moves all historical tensors to the target device.
    pub fn to_device(self, device: &B::Device) -> Self {
        Self {
            history_s: self
                .history_s
                .into_iter()
                .map(|t| t.to_device(device))
                .collect(),
            history_y: self
                .history_y
                .into_iter()
                .map(|t| t.to_device(device))
                .collect(),
        }
    }
}
impl<B: Backend> Default for LBFGSState<B> {
    fn default() -> Self {
        Self {
            history_s: Vec::new(),
            history_y: Vec::new(),
        }
    }
}

/// L-BFGS optimizer.
#[derive(Clone)]
pub struct LBFGS<B: Backend + AutodiffBackend> {
    config: LBFGSConfig,
    state: LBFGSState<B::InnerBackend>,
}

impl<B: Backend + AutodiffBackend> LBFGS<B> {
    /// A single optimization step for any tensor that represents the parameters of a model.
    pub fn step<M, F>(&mut self, lr: LearningRate, mut module: M, mut closure: F) -> (M, f64)
    where
        M: AutodiffModule<B> + Clone,
        F: FnMut(M) -> (f64, GradientsParams),
    {
        // evaluate initial f(x) and df/dx
        let (mut loss, grads) = closure(module.clone());
        let mut current_evals = 1;

        let mut flat_grad = flatten_grads_inner::<B, M>(&module, &grads);
        let mut x_flat = flatten_params_inner::<B, M>(&module);

        let opt_cond =
            flat_grad.clone().abs().max().into_scalar().to_f64() <= self.config.tolerance_grad;
        // optimal condition
        if opt_cond {
            return (module, loss);
        }

        // tensors cached in state (for tracing)
        let mut d = flat_grad.clone().neg();
        let mut t = lr;
        let mut prev_flat_grad: Option<Tensor<B::InnerBackend, 1>> = None;
        let mut n_iter = 0;

        // optimize for a max of max_iter iterations
        while n_iter < self.config.max_iter {
            // keep track of nb of iterations
            n_iter += 1;

            // compute gradient descent direction
            if n_iter == 1 {
                d = flat_grad.clone().neg();
                self.state.history_s.clear();
                self.state.history_y.clear();
            } else {
                // do lbfgs update (update memory)
                if let Some(pg) = prev_flat_grad.as_ref() {
                    let y = flat_grad.clone().sub(pg.clone());
                    let s = d.clone().mul_scalar(t);

                    let ys = y.clone().dot(s.clone()).into_scalar().to_f64();

                    if ys > 1e-10 {
                        // updating memory
                        if self.state.history_s.len() >= self.config.history_size {
                            // shift hisotry by one (limited-memory)
                            self.state.history_s.remove(0);
                            self.state.history_y.remove(0);
                        }
                        self.state.history_s.push(s);
                        self.state.history_y.push(y);
                    }
                }

                // compute the approximate (L-BFGS) inverse Hessian
                // multiplied by the gradient
                let num_old = self.state.history_s.len();
                let mut q = flat_grad.clone().neg();
                let mut alphas = vec![0.0; num_old];

                if num_old > 0 {
                    // multiply by initial Hessian
                    // r/d is the final direction
                    for i in (0..num_old).rev() {
                        let s = &self.state.history_s[i];
                        let y = &self.state.history_y[i];
                        let rho = 1.0 / y.clone().dot(s.clone()).into_scalar().to_f64();
                        alphas[i] = rho * s.clone().dot(q.clone()).into_scalar().to_f64();
                        q = q.sub(y.clone().mul_scalar(alphas[i]));
                    }

                    let last_s = &self.state.history_s[num_old - 1];
                    let last_y = &self.state.history_y[num_old - 1];
                    let ys = last_y.clone().dot(last_s.clone()).into_scalar().to_f64();
                    let yy = last_y.clone().dot(last_y.clone()).into_scalar().to_f64();
                    let h_diag = ys / yy;

                    let mut r = q.mul_scalar(h_diag);

                    for ((s, y), &alpha) in self
                        .state
                        .history_s
                        .iter()
                        .zip(self.state.history_y.iter())
                        .zip(alphas.iter())
                        .take(num_old)
                    {
                        let rho = 1.0 / y.clone().dot(s.clone()).into_scalar().to_f64();
                        let beta = rho * y.clone().dot(r.clone()).into_scalar().to_f64();
                        r = r.add(s.clone().mul_scalar(alpha - beta));
                    }
                    d = r;
                } else {
                    d = q;
                }
            }

            prev_flat_grad = Some(flat_grad.clone());
            let prev_loss = loss;

            // compute step len
            if n_iter == 1 {
                let grad_l1 = flat_grad.clone().abs().sum().into_scalar().to_f64();
                t = (1.0f64 / grad_l1).min(1.0) * lr;
            } else {
                t = lr;
            }

            // directional derivative
            let gtd = flat_grad.clone().dot(d.clone()).into_scalar().to_f64();

            if gtd > -self.config.tolerance_change {
                break;
            }

            let ls_func_evals;

            if let LineSearchFn::StrongWolfe = self.config.line_search_fn {
                // perform line search, using user function
                let mut obj_func =
                    |current_x: &Tensor<B::InnerBackend, 1>,
                     step: f64,
                     dir: &Tensor<B::InnerBackend, 1>| {
                        let update = dir.clone().mul_scalar(step);
                        let new_x = current_x.clone().add(update);
                        let tmp_module = set_params_from_flat_inner::<B, M>(module.clone(), new_x);
                        let (l, g) = closure(tmp_module);
                        (l, flatten_grads_inner::<B, M>(&module, &g))
                    };

                let (ls_f, ls_g, ls_t, evals) = strong_wolfe(
                    &mut obj_func,
                    &x_flat,
                    t,
                    &d,
                    loss,
                    flat_grad.clone(),
                    gtd,
                    1e-4,
                    0.9,
                    self.config.tolerance_change,
                    self.config.max_eval.unwrap() - current_evals,
                );

                loss = ls_f;
                flat_grad = ls_g;
                t = ls_t;
                ls_func_evals = evals;

                x_flat = x_flat.add(d.clone().mul_scalar(t));
                module = set_params_from_flat_inner::<B, M>(module, x_flat.clone());
            } else {
                // no line search, simply move with fixed-step
                let step_vec = d.clone().mul_scalar(t);
                x_flat = x_flat.add(step_vec);
                module = set_params_from_flat_inner::<B, M>(module, x_flat.clone());
                // re-evaluate function only if not in last iteration
                // the reason we do this: in a stochastic setting,
                // no use to re-evaluate that function here
                let (new_loss, new_grads) = closure(module.clone());
                loss = new_loss;
                flat_grad = flatten_grads_inner::<B, M>(&module, &new_grads);
                ls_func_evals = 1;
            }

            // update func eval
            current_evals += ls_func_evals;

            // check conditions

            if current_evals >= self.config.max_eval.unwrap() {
                break;
            }

            if flat_grad.clone().abs().max().into_scalar().to_f64() <= self.config.tolerance_grad {
                break;
            }

            if d.clone().mul_scalar(t).abs().max().into_scalar().to_f64()
                <= self.config.tolerance_change
            {
                break;
            }

            if (loss - prev_loss).abs() < self.config.tolerance_change {
                break;
            }
        }

        (module, loss)
    }
    /// Moves the optimizer state to the specified device.
    pub fn to_device(self, device: &B::Device) -> Self {
        Self {
            config: self.config,
            // History tensors reside in InnerBackend, so we convert the device accordingly
            state: self.state.to_device(device),
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::GradientsParams;
    use crate::TestAutodiffBackend;
    use burn::module::{Module, Param};
    use burn::tensor::{Tensor, TensorData};
    use burn_nn::{Linear, LinearConfig, LinearRecord};

    fn given_linear_layer(weight: TensorData, bias: TensorData) -> Linear<TestAutodiffBackend> {
        let device = Default::default();
        let record = LinearRecord {
            weight: Param::from_data(weight, &device),
            bias: Some(Param::from_data(bias, &device)),
        };

        LinearConfig::new(6, 6).init(&device).load_record(record)
    }
    #[test]
    fn test_cubic_interpolate() {
        let tolerance = 1e-8;

        // basic
        let (x1, f1, g1, x2, f2, g2) = (-1.0, 1.0, -2.0, 1.0, 1.0, 2.0);
        let result = cubic_interpolate(x1, f1, g1, x2, f2, g2, None);
        assert!(
            (result - 0.00000).abs() < tolerance,
            "Basic: Result {} should be close to 0.0",
            result
        );

        // bound
        let (x1, f1, g1, x2, f2, g2) = (0.0, 0.25, -1.0, 1.0, 0.25, 1.0);
        let bounds = Some((0.6, 1.0));
        let result = cubic_interpolate(x1, f1, g1, x2, f2, g2, bounds);
        assert!(
            (result - 0.6000000000).abs() < tolerance,
            "Bound: Result {} should be clamped to 0.6",
            result
        );

        // d2_square < 0,should return mid value
        let (x1, f1, g1, x2, f2, g2) = (0.0, 0.0, 10.0, 1.0, 5.0, 10.0);
        let result = cubic_interpolate(x1, f1, g1, x2, f2, g2, Some((0.0, 1.0)));
        assert!(
            (result - 0.5000000).abs() < tolerance,
            "Fallback: Result {} should be midpoint 0.5",
            result
        );

        // asymmetric
        let (x1, f1, g1, x2, f2, g2) = (0.0, 1.0, -5.0, 1.0, 0.5, 1.0);
        let result = cubic_interpolate(x1, f1, g1, x2, f2, g2, None);
        assert!(
            (result - 0.4606553370833684).abs() < tolerance,
            "Asymmetric: Result {} should be 0.4606553370833684",
            result
        );

        // not good value
        let (x1, f1, g1, x2, f2, g2) = (
            1.231232145,
            -0.12567458754,
            9.1231243007,
            8.239105015,
            -100.9012398021,
            123201321.0293982,
        );
        let result_1 = cubic_interpolate(x1, f1, g1, x2, f2, g2, None);
        let result_2 = cubic_interpolate(x1, f1, g1, x2, f2, g2, Some((-4.4, 4.4)));
        assert!(
            (result_1 - 5.9031480234724434).abs() < tolerance,
            "not good value 1: Result {} should be 5.9031480234724434",
            result
        );
        assert!(
            (result_2 - 4.4000000000000004).abs() < tolerance,
            "not good value 2: Result {} should be 4.4000000000000004",
            result
        );
    }
    #[test]
    fn test_strong_wolfe_direct_comparison() {
        let device = Default::default();
        let tol = 1e-8;

        {
            let x = Tensor::<TestAutodiffBackend, 1>::from_floats([2.1321912957_f64], &device);
            let d = Tensor::<TestAutodiffBackend, 1>::from_floats([0.91312321_f64], &device);
            let t_initial = 1.213132_f64;
            fn func<B: Backend>(
                x_base: &Tensor<B, 1>,
                t_val: f64,
                d_vec: &Tensor<B, 1>,
            ) -> (f64, Tensor<B, 1>) {
                let curr_x = x_base.clone().add(d_vec.clone().mul_scalar(t_val));
                let x2 = curr_x.clone().mul(curr_x.clone());
                let x3 = x2.clone().mul(curr_x.clone());
                let x4 = x2.clone().mul(x2.clone());

                // f(x) = x^4 - 2*x^2 + x
                let f_elements = x4 - x2.mul_scalar(2.0) + curr_x.clone();

                let f_val = f_elements.sum().into_scalar().to_f64();

                // g(x) = 4*x^3 - 4*x + 1
                let g = x3.mul_scalar(4.0) - curr_x.clone().mul_scalar(4.0)
                    + Tensor::ones_like(&curr_x);

                (f_val, g)
            }
            let (f_init, g_init) = func(&x, 0.0, &d);
            let gtd_init = g_init.clone().dot(d.clone()).into_scalar().to_f64();
            println!("Initial State: f={},gtd = {}", f_init, gtd_init);
            assert!((f_init - 13.7080059052).abs() < tol);
            assert!((gtd_init - 28.5305728912).abs() < tol);
            let mut obj_func =
                |xb: &Tensor<TestAutodiffBackend, 1>,
                 tv: f64,
                 dv: &Tensor<TestAutodiffBackend, 1>| func(xb, tv, dv);

            let (f_final, _g_final, t_final, evals) = strong_wolfe(
                &mut obj_func,
                &x,
                t_initial,
                &d,
                f_init,
                g_init,
                gtd_init,
                1e-4, // c1
                0.9,  // c2
                1e-9, // tolerance_change
                10,   // max_ls
            );
            let g_f = _g_final.into_scalar().to_f64();
            println!(
                "f_final:{:?},_g_final:{:?},t_final:{:?},evals:{:?}",
                f_final, g_f, t_final, evals
            );
            assert!((f_final - 13.708005905151367).abs() < tol);
            assert!((g_f - 31.2450428009).abs() < tol);
            assert!((t_final.to_f64() - 0.0).abs() < tol);
            assert!((evals == 11));
        }
    }
    #[test]
    fn test_lbfgs_strong_wolfe_comparison() {
        let device = Default::default();
        let tol = 1e-5;
        let x_data = Tensor::<TestAutodiffBackend, 2>::from_data([[1.0], [2.0], [3.0]], &device);
        let y_true = Tensor::<TestAutodiffBackend, 2>::from_data([[3.0], [5.0], [7.0]], &device);
        let weight = TensorData::from([[0.5f64]]);
        let bias = TensorData::from([0.1f64]);
        let module = given_linear_layer(weight, bias);

        let mut optimizer: LBFGS<TestAutodiffBackend> = LBFGSConfig::new()
            .with_line_search_fn(LineSearchFn::StrongWolfe)
            .init();
        let mut closure = |mod_in: Linear<TestAutodiffBackend>| {
            let output = mod_in.forward(x_data.clone());
            let loss = burn_nn::loss::MseLoss::new().forward(
                output,
                y_true.clone(),
                burn_nn::loss::Reduction::Sum,
            );

            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &mod_in);

            (loss.into_scalar().to_f64(), grads_params)
        };
        let initial_loss = closure(module.clone()).0;
        assert!((initial_loss - 50.1300048828).abs() < tol);
        let (updated_module, final_loss) = optimizer.step(0.001, module, &mut closure);
        assert!((final_loss - 0.0234732367).abs() < tol);
        let optimized_data: f64 = updated_module.weight.val().into_scalar().to_f64();
        let optimized_bias: f64 = updated_module
            .bias
            .as_ref()
            .unwrap()
            .val()
            .into_scalar()
            .to_f64();
        assert!((optimized_data - 2.0570652485).abs() < tol);
        assert!((optimized_bias - 0.8106800914).abs() < tol);
    }
    #[test]
    fn test_lbfgs_no_strong_wolfe_comparison() {
        let device = Default::default();
        let tol = 1e-5;
        let x_data = Tensor::<TestAutodiffBackend, 2>::from_data([[1.0], [2.0], [3.0]], &device);
        let y_true = Tensor::<TestAutodiffBackend, 2>::from_data([[3.0], [5.0], [7.0]], &device);
        let weight = TensorData::from([[0.5f64]]);
        let bias = TensorData::from([0.1f64]);
        let module = given_linear_layer(weight, bias);

        let mut optimizer: LBFGS<TestAutodiffBackend> = LBFGSConfig::new()
            .with_line_search_fn(LineSearchFn::None)
            .init();
        let mut closure = |mod_in: Linear<TestAutodiffBackend>| {
            let output = mod_in.forward(x_data.clone());
            let loss = burn_nn::loss::MseLoss::new().forward(
                output,
                y_true.clone(),
                burn_nn::loss::Reduction::Sum,
            );

            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &mod_in);

            (loss.into_scalar().to_f64(), grads_params)
        };
        let initial_loss = closure(module.clone()).0;
        assert!((initial_loss - 50.1300048828).abs() < tol);
        let (updated_module, final_loss) = optimizer.step(0.001, module, &mut closure);
        assert!((final_loss - 48.2181930542).abs() < tol);
        let optimized_data: f64 = updated_module.weight.val().into_scalar().to_f64();
        let optimized_bias: f64 = updated_module
            .bias
            .as_ref()
            .unwrap()
            .val()
            .into_scalar()
            .to_f64();

        assert!((optimized_data - 0.5302446192).abs() < tol);
        assert!((optimized_bias - 0.1142520783).abs() < tol);
    }
}
