use super::kernel::{TOUCHED_PER_WRITTEN, touch_vectors};
use cubecl::calculate_cube_count_elemwise;
use cubecl::client::Client;
use cubecl::config::{CubeClRuntimeConfig, RuntimeConfig};
use cubecl::ir::{ElemType, FloatKind};
use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;
use cubecl::std::tensor::layout::linear::linear_view;
use cubecl::std::throughput::{compute_bound, measure_launch_overhead, memory_bound};
use cubecl::tensor_vector_size_parallel;
use cubecl::throughput::{MemoryAccess, ThroughputKey};
use cubecl::tune::{
    AutotuneError, Bounds, BoundsGenerator, Eviction, Thresholds, TuneInputs, Work,
};
use cubecl::zspace::Shape;
use std::sync::Arc;

/// What a problem costs on the device it runs on — the roofline a round
/// measures its candidates against, stated by the family in
/// [`Autotune::bounded`](super::Autotune::bounded).
pub struct Roofline {
    /// The device, which is where the peak is measured.
    pub client: Client,
    /// The resource the family's candidates are bound by, whose peak is the
    /// roofline.
    pub ceiling: Ceiling,
    /// The operations and the bytes the problem needs at minimum.
    pub work: Work,
}

impl Roofline {
    /// Read past the last-level cache before a sample of this problem, where the problem is
    /// small enough to have been served from it, wholly or in part: one that moves less than a
    /// read past the cache covers ([`cold_read_bytes`]). Over that many bytes ([`touch`]), so no
    /// sample sees its own warm-up. A problem as large as the cache itself is not past it: its
    /// next sample still finds much of it there.
    fn read_cold(&self) {
        let bytes = cold_read_bytes(&self.client);
        if self.work.bytes >= bytes {
            return;
        }
        touch(&self.client, bytes);
    }
}

/// The resource a family's candidates are bound by. A family says which, rather
/// than the autotune taking the slower of the two halves by default: an autotune
/// that measured both would probe a peak its candidates never approach, and a
/// stop is only ever argued from the resource the kernel actually saturates.
///
/// A family whose problems straddle the machine's balance names
/// [`Either`](Self::Either) and says so: there the wall a key runs into is
/// the key's, not the family's, and the stop is argued from whichever is
/// higher for that key.
pub enum Ceiling {
    /// The instruction the arithmetic runs on — prefill attention, whose
    /// every key reuses each key and value across a whole query block.
    Compute(ThroughputKey),
    /// The bus, in the direction the traffic that dominates goes — a product of
    /// few rows, which streams far more than it writes and so reaches past a
    /// copy's bandwidth; a bound at the copy would let a candidate at a
    /// fraction of the bus pass for close enough.
    Memory(MemoryAccess),
    /// Whichever of the two is higher **for the key**: the floor is the slower
    /// wall, and a round stops at a candidate within the level's fraction of
    /// that.
    ///
    /// For a family whose intensity crosses the machine's balance inside the
    /// shapes it serves — the matmul at prefill widths, where a projection's
    /// flops per byte sit either side of the device's — a single wall is the
    /// wrong one for half the keys, and wrong in the direction that matters:
    /// a compute wall on a memory-bound key states a floor *below* the bus's,
    /// which no candidate can reach, so the round never stops and compiles the
    /// whole list.
    Either {
        /// The instruction the arithmetic runs on.
        compute: ThroughputKey,
        /// The direction of the traffic that dominates.
        memory: MemoryAccess,
    },
}

/// What a read past the last-level cache covers where the runtime reports no
/// cache: twice the largest a device in this class carries, the 128 MB of an
/// RTX PRO 6000's L2, as [`cold_read_bytes`] reads twice a reported one. CUDA,
/// HIP, wgpu and Metal report none, so on a GPU this is the figure.
pub const COLD_READ: usize = 256 << 20;

/// Bytes a read past `client`'s last-level cache covers: twice the cache the runtime reports, or
/// [`COLD_READ`] where it reports none. What a measurement reads before a sample the cache would
/// otherwise serve.
pub fn cold_read_bytes(client: &Client) -> usize {
    client
        .properties()
        .hardware
        .last_level_cache_size
        .map_or(COLD_READ, |cache| 2 * cache)
}

/// Read `bytes` of scratch through the device's caches, writing back one
/// vector per `TOUCHED_PER_WRITTEN` read: what a measurement runs before a
/// sample whose operands the last-level cache would otherwise serve. A read
/// and not a fill, because a fill leaves the cache full of dirty data that
/// writes back **during** the sample it was meant to clean, and a
/// memory-bound sample then pays that traffic as its own.
pub fn touch(client: &Client, bytes: usize) {
    let elem = ElemType::Float(FloatKind::F32);
    let scratch = TensorHandle::empty(client, Shape::new([bytes / elem.size()]), elem);
    let _ = read_through(client, scratch);
}

/// Read every vector of `scratch` through the device's caches, once, and hand back the sum each
/// unit wrote: [`touch`]'s read over a buffer the caller filled, so that the read covers the
/// whole of it is a sum a test checks rather than a stopwatch a device could flatter.
///
/// Vector `j` of the result is the sum of vectors `j + i · written` of `scratch` for `i` below
/// `TOUCHED_PER_WRITTEN`, `written` being the result's own length in vectors.
pub fn read_through(client: &Client, scratch: TensorHandle) -> TensorHandle {
    let elem = scratch.dtype;
    let vector_size = tensor_vector_size_parallel(
        client.io_optimized_vector_sizes(elem.size()),
        scratch.shape(),
        scratch.strides(),
        0,
    );
    let elements = scratch.shape().num_elements();
    let written = (elements / vector_size).div_ceil(TOUCHED_PER_WRITTEN as usize) * vector_size;
    let output = TensorHandle::empty(client, Shape::new([written]), elem);
    let working_units = written / vector_size;
    let cube_dim = CubeDim::new(client, working_units);
    let cube_count = calculate_cube_count_elemwise(client, working_units, cube_dim);
    touch_vectors::launch(
        client,
        cube_count,
        cube_dim,
        scratch
            .required_address_type()
            .max(output.required_address_type()),
        vector_size,
        linear_view(scratch.binding()),
        linear_view(output.clone().binding()),
        elem,
    );
    output
}

/// A family's [`Roofline`] as cubecl's eviction: [`Roofline::read_cold`] before every
/// measured sample of a problem the cache would otherwise serve.
pub(super) struct ReadCold<F>(pub(super) Arc<F>);

impl<K, I, F> Eviction<K, I> for ReadCold<F>
where
    K: 'static,
    I: TuneInputs,
    F: for<'a> Fn(&K, &<I as TuneInputs>::At<'a>) -> Roofline + Send + Sync + 'static,
{
    fn evict<'a>(&self, key: &K, inputs: &<I as TuneInputs>::At<'a>) -> Result<(), AutotuneError> {
        (self.0)(key, inputs).read_cold();
        Ok(())
    }
}

/// A family's [`Roofline`] as cubecl's bounds generator: the roofline at the
/// configured level's threshold. A level that measures everything states an
/// unbounded threshold, which is a bound that never gives a time limit.
pub(super) struct Bounded<F>(pub(super) Arc<F>);

impl<K, I, F> BoundsGenerator<K, I> for Bounded<F>
where
    K: 'static,
    I: TuneInputs,
    F: for<'a> Fn(&K, &<I as TuneInputs>::At<'a>) -> Roofline + Send + Sync + 'static,
{
    fn generate<'a>(&self, key: &K, inputs: &<I as TuneInputs>::At<'a>) -> Bounds {
        // Read per tune rather than once, so a level set after the autotune was
        // built still reaches it.
        let thresholds = Thresholds::for_level(&CubeClRuntimeConfig::get().autotune.level);
        let Roofline {
            client,
            ceiling,
            work,
        } = (self.0)(key, inputs);
        // cubecl reads a list of bounds as the slowest of them — the time
        // limit is the longest — which is what lets a family name both walls
        // and have the key pick.
        let bounds = match ceiling {
            Ceiling::Compute(instruction) => {
                vec![compute_bound(
                    &client,
                    instruction,
                    work,
                    thresholds.compute,
                )]
            }
            Ceiling::Memory(access) => vec![memory_bound(&client, access, work, thresholds.memory)],
            Ceiling::Either { compute, memory } => vec![
                compute_bound(&client, compute, work, thresholds.compute),
                memory_bound(&client, memory, work, thresholds.memory),
            ],
        };
        Bounds {
            bounds,
            launch_overhead: measure_launch_overhead(&client),
        }
    }
}
