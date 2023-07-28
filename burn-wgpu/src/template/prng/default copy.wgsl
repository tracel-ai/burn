// Define PRNG parameters
const kLCGMultiplier : u32 = 1664525u;
const kLCGIncrement : u32 = 1013904223u;
const kTauswortheBitMask : u32 = 0x000FFFFFu;

// Define the PRNG state
struct PRNGState {
    lcgState: u32,
    tauswortheState: u32,
}

// Function to initialize the PRNG state
fn initPRNGState(seed: u32) -> PRNGState {
    let prng_state = PRNGState (seed, seed ^ (seed >> 7));
    return prng_state;
}

// Function to update the PRNG state
fn updatePRNGState(state: PRNGState) {
    // Update the LCG state
    state.lcgState = kLCGMultiplier * state.lcgState + kLCGIncrement;

    // Update the Tausworthe state
    let newBit = ((state.tauswortheState << 13) ^ state.tauswortheState) & kTauswortheBitMask;
    state.tauswortheState = ((state.tauswortheState & 0xfffffffeu) << 18) ^ newBit;
}

struct PRNGBuffer {
    data: [[stride(8)]] array<PRNGState>;
};

struct RandomNumberBuffer {
    data: [[stride(4)]] array<f32>;
};

[[group(0), binding(0)]]
var<storage, read_write> prngBuffer : PRNGBuffer;

[[group(0), binding(1)]]
var<storage, write> randomNumberBuffer : RandomNumberBuffer;

@compute
@workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let index = global_id.y * (num_workgroups.x * {{ workgroup_size_x }}u) + global_id.x;
    var prngState : PRNGState = prngBuffer.data[index];

    // Generate the random number for the current element
    var randomNumber = f32(prngState.lcgState ^ prngState.tauswortheState) / 4294967296.0;
    randomNumberBuffer.data[index] = randomNumber;

    // Update the PRNG state for the next iteration
    updatePRNGState(prngState);
    prngBuffer.data[index] = prngState;
}
