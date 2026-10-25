use super::coordinate;
use burn_core::tensor::{DType, Element, ElementConversion};
use bytemuck::Pod;

/// Returns false unless every value is bitwise equal to an RGB8 level / 255.
pub(super) fn try_filter(
    input: &[f32],
    output: &mut [f32],
    height: usize,
    width: usize,
    size: usize,
) -> bool {
    let levels: [f32; 256] = core::array::from_fn(|i| i as f32 / 255.0);
    let encode = |value: f32| {
        let level = (value * 255.0).round() as usize;
        (level < levels.len() && levels[level].to_bits() == value.to_bits()).then_some(level as u8)
    };
    // Reject ordinary F32 images before allocating a plane of byte indices.
    if input.iter().take(64).any(|&value| encode(value).is_none()) {
        return false;
    }
    let Some(indices) = input
        .iter()
        .map(|&value| encode(value))
        .collect::<Option<Vec<_>>>()
    else {
        return false;
    };

    filter_levels(&indices, output, &levels, [height, width], size, 0);
    true
}

pub(super) fn try_integer<E: Element + Pod>(
    input: &[E],
    output: &mut [E],
    height: usize,
    width: usize,
    size: usize,
) -> bool {
    let xor = match E::dtype() {
        DType::U8 => 0,
        DType::I8 => 0x80,
        _ => return false,
    };
    let levels = core::array::from_fn(|i| (i as i16 - i16::from(xor)).elem::<E>());
    // Flipping the sign bit maps two's-complement I8 bytes to ascending bins.
    filter_levels(
        bytemuck::cast_slice(input),
        output,
        &levels,
        [height, width],
        size,
        xor,
    );
    true
}

fn filter_levels<E: Copy>(
    indices: &[u8],
    output: &mut [E],
    levels: &[E; 256],
    [height, width]: [usize; 2],
    size: usize,
    xor: u8,
) {
    let reach = size / 2;
    for y in 0..height {
        let mut histogram = Histogram::new(size * size / 2);
        for dy in 0..size {
            let sy = coordinate(y, dy, reach, height);
            for dx in 0..size {
                let sx = coordinate(0, dx, reach, width);
                histogram.add(indices[sy * width + sx] ^ xor);
            }
        }
        output[y * width] = levels[histogram.median()];
        for x in 1..width {
            let leaving = coordinate(x - 1, 0, reach, width);
            let entering = coordinate(x, size - 1, reach, width);
            for dy in 0..size {
                let row = coordinate(y, dy, reach, height) * width;
                histogram.remove(indices[row + leaving] ^ xor);
                histogram.add(indices[row + entering] ^ xor);
            }
            output[y * width + x] = levels[histogram.median()];
        }
    }
}

struct Histogram {
    counts: [usize; 256],
    rank: usize,
    median: usize,
    /// Number of samples strictly below the current median bin.
    below: usize,
}

impl Histogram {
    fn new(rank: usize) -> Self {
        Self {
            counts: [0; 256],
            rank,
            median: 0,
            below: 0,
        }
    }

    fn add(&mut self, value: u8) {
        let value = value as usize;
        self.counts[value] += 1;
        self.below += usize::from(value < self.median);
    }

    fn remove(&mut self, value: u8) {
        let value = value as usize;
        self.counts[value] -= 1;
        self.below -= usize::from(value < self.median);
    }

    fn median(&mut self) -> usize {
        while self.below > self.rank {
            self.median -= 1;
            self.below -= self.counts[self.median];
        }
        while self.below + self.counts[self.median] <= self.rank {
            self.below += self.counts[self.median];
            self.median += 1;
        }
        self.median
    }
}
