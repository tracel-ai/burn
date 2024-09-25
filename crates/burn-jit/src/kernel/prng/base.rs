use cubecl::{
    cpa,
    ir::{Elem, Scope, Variable},
    prelude::*,
    CubeCountSettings, Execution, InputInfo, OutputInfo,
};
use std::marker::PhantomData;

use crate::{
    kernel::{Kernel, SUBCUBE_DIM_APPROX},
    tensor::JitTensor,
    JitElement, JitRuntime, SEED,
};
use burn_common::rand::get_seeded_rng;
use burn_tensor::Shape;
use rand::Rng;

pub(crate) const N_VALUES_PER_THREAD: usize = 128;

/// Pseudo-random generator
pub(crate) fn random<P: Prng<E>, R: JitRuntime, E: JitElement>(
    shape: Shape,
    device: &R::Device,
    prng: P,
) -> JitTensor<R, E> {
    let client = R::client(device);
    let kernel: PrngEagerKernel<P, R, E> = PrngEagerKernel::new();
    let num_elems = shape.num_elements();
    let buffer = client.empty(num_elems * core::mem::size_of::<E>());
    let output = JitTensor::new_contiguous(client.clone(), device.clone(), shape.clone(), buffer);
    let seeds = get_seeds();

    Execution::start(kernel, client)
        .outputs(&[output.as_handle_ref()])
        .with_scalars(&seeds)
        .with_scalars(&prng.args())
        .execute(CubeCountSettings::Custom(prng_cube_count::<R>(
            num_elems,
            SUBCUBE_DIM_APPROX,
            N_VALUES_PER_THREAD,
        )));

    output
}

fn prng_cube_count<R: JitRuntime>(
    num_elems: usize,
    cube_dim: usize,
    n_values_per_thread: usize,
) -> CubeCount<R::Server> {
    let num_threads = f32::ceil(num_elems as f32 / n_values_per_thread as f32);
    let num_elems_per_cube = cube_dim * cube_dim;
    let num_invocations = f32::ceil(num_threads / num_elems_per_cube as f32);
    let cubes_x = f32::ceil(f32::sqrt(num_invocations));
    let cubes_y = f32::ceil(num_invocations / cubes_x);

    CubeCount::Static(cubes_x as u32, cubes_y as u32, 1)
}

impl<P: Prng<E>, R: JitRuntime, E: JitElement> Kernel for PrngEagerKernel<P, R, E> {
    fn define(&self) -> KernelDefinition {
        let mut scope = Scope::root();
        let item = E::cube_elem().into();

        let output = Variable::GlobalOutputArray { id: 0, item };

        let seed0 = Variable::GlobalScalar {
            id: 0,
            elem: Elem::UInt,
        };
        let seed1 = Variable::GlobalScalar {
            id: 1,
            elem: Elem::UInt,
        };
        let seed2 = Variable::GlobalScalar {
            id: 2,
            elem: Elem::UInt,
        };
        let seed3 = Variable::GlobalScalar {
            id: 3,
            elem: Elem::UInt,
        };
        let seeds = [seed0, seed1, seed2, seed3];

        let mut args = Vec::<Variable>::new();
        for i in 0..P::args_length() {
            args.push(Variable::GlobalScalar {
                id: i as u16,
                elem: item.elem(),
            });
        }

        PrngShader::<P, E>::new(output, N_VALUES_PER_THREAD, seeds, args).expand(&mut scope);

        scope.write_global_custom(output);

        let args = InputInfo::Scalar {
            elem: E::cube_elem(),
            size: P::args_length(),
        };
        let seeds = InputInfo::Scalar {
            elem: Elem::UInt,
            size: 4,
        };
        let out = OutputInfo::Array { item };

        let info = KernelExpansion {
            inputs: vec![args, seeds],
            outputs: vec![out],
            scope,
        };

        let settings = KernelSettings::default();
        KernelIntegrator::new(info).integrate(settings)
    }

    fn id(&self) -> cubecl::KernelId {
        cubecl::KernelId::new::<Self>()
    }
}

#[derive(new)]
pub(crate) struct PrngEagerKernel<P: Prng<E>, R: JitRuntime, E: JitElement> {
    _prng: PhantomData<P>,
    _runtime: PhantomData<R>,
    _elem: PhantomData<E>,
}

pub(crate) fn get_seeds() -> [u32; 4] {
    let mut seed = SEED.lock().unwrap();
    let mut rng = match seed.as_ref() {
        Some(rng_seeded) => rng_seeded.clone(),
        None => get_seeded_rng(),
    };
    let mut seeds: Vec<u32> = Vec::with_capacity(4);
    for _ in 0..4 {
        seeds.push(rng.gen());
    }
    *seed = Some(rng);

    seeds.try_into().unwrap()
}

pub(crate) trait Prng<E>: Send + Sync + 'static {
    fn args(self) -> Vec<E>;

    fn args_length() -> usize;

    #[allow(clippy::too_many_arguments)]
    fn inner_loop(
        scope: &mut Scope,
        args: Vec<Variable>,
        write_index_base: Variable,
        n_invocations: Variable,
        n_values_per_thread: usize,
        state_0: Variable,
        state_1: Variable,
        state_2: Variable,
        state_3: Variable,
        output: Variable,
    );
}

#[derive(new)]
pub(crate) struct PrngShader<P: Prng<E>, E: JitElement> {
    output: Variable,
    n_values_per_thread: usize,
    seeds: [Variable; 4],
    args: Vec<Variable>,
    _prng: PhantomData<P>,
    _elem: PhantomData<E>,
}

impl<P: Prng<E>, E: JitElement> PrngShader<P, E> {
    pub(crate) fn expand(self, scope: &mut Scope) {
        let output = self.output;
        let [seed_0, seed_1, seed_2, seed_3] = self.seeds;
        let n_values_per_thread: Variable = self.n_values_per_thread.into();
        let args = self.args;

        let cube_dim_x = Variable::CubeDimX;
        let cube_dim_y = Variable::CubeDimY;
        let cube_pos_x = Variable::CubePosX;
        let cube_pos_y = Variable::CubePosY;
        let cube_count_y = Variable::CubeCountY;
        let local_index = Variable::UnitPos;

        let n_invocations = scope.create_local(Elem::UInt);
        cpa!(scope, n_invocations = cube_dim_x);
        cpa!(scope, n_invocations *= cube_dim_y);

        let cube_offset = scope.create_local(Elem::UInt);
        cpa!(scope, cube_offset = cube_pos_x * cube_count_y);
        cpa!(scope, cube_offset += cube_pos_y);
        cpa!(scope, cube_offset *= n_invocations);

        let write_index_base = scope.create_local(Elem::UInt);
        cpa!(scope, write_index_base = cube_offset);
        cpa!(scope, write_index_base *= n_values_per_thread);
        cpa!(scope, write_index_base += local_index);

        // Set state with unique seeds
        let thread_seed = scope.create_local(Elem::UInt);
        cpa!(scope, thread_seed = cast(1000000007));
        let thread_seed_index = scope.create_local(Elem::UInt);
        cpa!(scope, thread_seed_index = cube_offset + local_index);
        cpa!(scope, thread_seed *= thread_seed_index);

        let state_0 = scope.create_local(Elem::UInt);
        cpa!(scope, state_0 = thread_seed);
        cpa!(scope, state_0 += seed_0);

        let state_1 = scope.create_local(Elem::UInt);
        cpa!(scope, state_1 = thread_seed);
        cpa!(scope, state_1 += seed_1);

        let state_2 = scope.create_local(Elem::UInt);
        cpa!(scope, state_2 = thread_seed);
        cpa!(scope, state_2 += seed_2);

        let state_3 = scope.create_local(Elem::UInt);
        cpa!(scope, state_3 = thread_seed);
        cpa!(scope, state_3 += seed_3);

        // Creation of n_values_per_thread values, specific to the distribution
        P::inner_loop(
            scope,
            args,
            write_index_base,
            n_invocations,
            self.n_values_per_thread,
            state_0,
            state_1,
            state_2,
            state_3,
            output,
        );
    }
}

pub(crate) fn taus_step_0(scope: &mut Scope, z: Variable) {
    taus_step(
        scope,
        z,
        13u32.into(),
        19u32.into(),
        12u32.into(),
        4294967294u32.into(),
    );
}

pub(crate) fn taus_step_1(scope: &mut Scope, z: Variable) {
    taus_step(
        scope,
        z,
        2u32.into(),
        25u32.into(),
        4u32.into(),
        4294967288u32.into(),
    );
}

pub(crate) fn taus_step_2(scope: &mut Scope, z: Variable) {
    taus_step(
        scope,
        z,
        3u32.into(),
        11u32.into(),
        17u32.into(),
        4294967280u32.into(),
    );
}

fn taus_step(
    scope: &mut Scope,
    z: Variable,
    s1: Variable,
    s2: Variable,
    s3: Variable,
    m: Variable,
) {
    let b = scope.create_local(Elem::UInt);
    cpa!(scope, b = z << s1);
    cpa!(scope, b = b ^ z);
    cpa!(scope, b = b >> s2);
    cpa!(scope, z = z & m);
    cpa!(scope, z = z << s3);
    cpa!(scope, z = z ^ b);
}

pub(crate) fn lcg_step(scope: &mut Scope, z: Variable) {
    let a: Variable = 1664525u32.into();
    let b: Variable = 1013904223u32.into();
    cpa!(scope, z *= a);
    cpa!(scope, z += b);
}

pub(crate) fn cast_uint_to_float(scope: &mut Scope, int_random: Variable, float_random: Variable) {
    let tmp: Variable = 2.328_306_4e-10f32.into();
    cpa!(scope, float_random = cast(int_random));
    cpa!(scope, float_random *= tmp);
}

#[allow(missing_docs)]
pub mod tests_utils {
    use burn_tensor::Element;

    #[derive(Default, Copy, Clone)]
    pub struct BinStats {
        pub count: usize,
        pub n_runs: usize, // Number of sequences of same bin
    }

    #[allow(unused)]
    pub fn calculate_bin_stats<E: Element>(
        numbers: &[E],
        number_of_bins: usize,
        low: f32,
        high: f32,
    ) -> Vec<BinStats> {
        let range = (high - low) / number_of_bins as f32;
        let mut output: Vec<BinStats> = (0..number_of_bins).map(|_| Default::default()).collect();
        let mut initialized = false;
        let mut current_runs = number_of_bins; // impossible value for starting point
        for number in numbers {
            let num = number.elem::<f32>();
            if num < low || num > high {
                continue;
            }
            let index = f32::floor((num - low) / range) as usize;
            output[index].count += 1;
            if initialized && index != current_runs {
                output[current_runs].n_runs += 1;
            }
            initialized = true;
            current_runs = index;
        }
        output[current_runs].n_runs += 1;
        output
    }
}
