use std::marker::PhantomData;

use crate::{
    gpu::{gpu, Elem, Scope, Variable},
    JitElement, Runtime, SEED,
};
use burn_common::rand::get_seeded_rng;
use rand::Rng;

pub(crate) const N_VALUES_PER_THREAD: usize = 128;

#[derive(new)]
pub(crate) struct PrngEagerKernel<P: Prng, R: Runtime, E: JitElement> {
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

pub(crate) trait Prng: Send + Sync + 'static {
    fn inner_loop(
        &self,
        scope: &mut Scope,
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
pub(crate) struct PrngShader<P: Prng> {
    output: Variable,
    n_values_per_thread: usize,
    seeds: [Variable; 4],
    prng: P,
}

impl<P: Prng> PrngShader<P> {
    pub(crate) fn expand(self, scope: &mut Scope) {
        let output = self.output;
        let [seed_0, seed_1, seed_2, seed_3] = self.seeds;
        let n_values_per_thread: Variable = self.n_values_per_thread.into();

        let workgroup_size_x = Variable::WorkgroupSizeX;
        let workgroup_size_y = Variable::WorkgroupSizeY;
        let workgroup_id_x = Variable::WorkgroupIdX;
        let workgroup_id_y = Variable::WorkgroupIdY;
        let num_workgroups_y = Variable::NumWorkgroupsY;
        let local_index = Variable::LocalInvocationIndex;

        let n_invocations = scope.create_local(Elem::UInt);
        gpu!(scope, n_invocations = workgroup_size_x);
        gpu!(scope, n_invocations *= workgroup_size_y);

        let workgroup_offset = scope.create_local(Elem::UInt);
        gpu!(scope, workgroup_offset = workgroup_id_x * num_workgroups_y);
        gpu!(scope, workgroup_offset += workgroup_id_y);
        gpu!(scope, workgroup_offset *= n_invocations);

        let write_index_base = scope.create_local(Elem::UInt);
        gpu!(scope, write_index_base = workgroup_offset);
        gpu!(scope, write_index_base *= n_values_per_thread);
        gpu!(scope, write_index_base += local_index);

        // Set state with unique seeds
        let thread_seed = scope.create_local(Elem::UInt);
        gpu!(scope, thread_seed = cast(1000000007));
        let thread_seed_index = scope.create_local(Elem::UInt);
        gpu!(scope, thread_seed_index = workgroup_offset + local_index);
        gpu!(scope, thread_seed *= thread_seed_index);

        let state_0 = scope.create_local(Elem::UInt);
        gpu!(scope, state_0 = thread_seed);
        gpu!(scope, state_0 += seed_0);

        let state_1 = scope.create_local(Elem::UInt);
        gpu!(scope, state_1 = thread_seed);
        gpu!(scope, state_1 += seed_1);

        let state_2 = scope.create_local(Elem::UInt);
        gpu!(scope, state_2 = thread_seed);
        gpu!(scope, state_2 += seed_2);

        let state_3 = scope.create_local(Elem::UInt);
        gpu!(scope, state_3 = thread_seed);
        gpu!(scope, state_3 += seed_3);

        // Creation of n_values_per_thread values, specific to the distribution
        self.prng.inner_loop(
            scope,
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
    gpu!(scope, b = z << s1);
    gpu!(scope, b = b ^ z);
    gpu!(scope, b = b >> s2);
    gpu!(scope, z = z & m);
    gpu!(scope, z = z << s3);
    gpu!(scope, z = z ^ b);
}

pub(crate) fn lcg_step(scope: &mut Scope, z: Variable) {
    let a: Variable = 1664525u32.into();
    let b: Variable = 1013904223u32.into();
    gpu!(scope, z *= a);
    gpu!(scope, z += b);
}

pub(crate) fn cast_uint_to_float(scope: &mut Scope, int_random: Variable, float_random: Variable) {
    let tmp: Variable = 2.3283064365387e-10.into();
    gpu!(scope, float_random = cast(int_random));
    gpu!(scope, float_random *= tmp);
}

#[cfg(feature = "export_tests")]
#[allow(missing_docs)]
pub mod tests_utils {
    use burn_tensor::Element;

    #[derive(Default, Copy, Clone)]
    pub struct BinStats {
        pub count: usize,
        pub n_runs: usize, // Number of sequences of same bin
    }

    pub fn calculate_bin_stats<E: Element>(
        numbers: Vec<E>,
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
