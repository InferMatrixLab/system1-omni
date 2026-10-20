//! Encoder execution for the admitted 0.3B profile against the
//! regenerated CUDA bundle. Weight upload, the YaRN rotary tables and
//! the workspace follow the family conventions; the executed step
//! sequence is the family core plan with this profile's `_h768`
//! kernel symbols. The final hidden of the last layer lives in the
//! bundle's bf16 working precision after the final norm; readouts
//! gather requested positions from it.

use anyhow::{Context, Result, ensure};
use half::bf16;
use omni_cuda::{Buffer, Cuda, Kernel, Ptr};
use omni_modernbert::geometry::EncoderScratch;
use omni_modernbert::plan::{self, EncoderPlanOptions, PlanArg, PlanStep, ScratchSlot};
use omni_modernbert::structure::ModernBertStructure;
use std::{collections::HashMap, path::Path};

use crate::config::read_structure;
use crate::profile;
use crate::rope::cos_sin_tables;
use crate::weights::Weights;

/// Plan kernel name → bundle symbol for the 768-hidden profile.
fn symbol(kernel: &str) -> &str {
    match kernel {
        "embed" => "embed_h768",
        "qkv" => "qkv_h768",
        "rope" => "rope_h768",
        "attn_full" => "attn_full_h768",
        "attn_local" => "attn_local_h768",
        "out" => "out_h768",
        "addln" => "addln_h768",
        "geglu" => "geglu_h768",
        "down" => "down_h768",
        other => other,
    }
}

#[derive(Clone, Copy)]
enum Arg {
    Weight(Ptr),
    Scratch(ScratchSlot),
}

struct Step {
    kernel: Kernel,
    args: [Arg; 5],
    count: usize,
}

struct Scratch {
    ids: Buffer,
    lens: Buffer,
    x: Buffer,
    y: Buffer,
    qkv: Buffer,
    o: Buffer,
    gate: Buffer,
}

impl Scratch {
    fn new(cuda: &Cuda, structure: &ModernBertStructure, batch: usize, len: usize) -> Result<Self> {
        let sizes = EncoderScratch::new(batch, len, structure);
        let mut values = sizes.sizes().into_iter();
        let mut alloc = || {
            cuda.alloc(values.next().expect("scratch layout"))
                .context("allocate encoder scratch")
        };
        Ok(Self {
            ids: alloc()?,
            lens: alloc()?,
            x: alloc()?,
            y: alloc()?,
            qkv: alloc()?,
            o: alloc()?,
            gate: alloc()?,
        })
    }

    fn buffer(&self, slot: ScratchSlot) -> &Buffer {
        match slot {
            ScratchSlot::Ids => &self.ids,
            ScratchSlot::Lens => &self.lens,
            ScratchSlot::Types => &self.ids,
            ScratchSlot::X => &self.x,
            ScratchSlot::Y => &self.y,
            ScratchSlot::Qkv => &self.qkv,
            ScratchSlot::O => &self.o,
            ScratchSlot::Gate => &self.gate,
        }
    }
}

pub struct Encoder {
    pub structure: ModernBertStructure,
    cuda: Cuda,
    /// Owns every uploaded tensor; the plan holds raw pointers into it.
    _device_tensors: HashMap<String, Buffer>,
    plan: Vec<Step>,
}

fn bytes16(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn bytes32(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

impl Encoder {
    /// Load the pinned bundle (config.json + model.safetensors) and the
    /// regenerated CUDA library, admitting the structure first.
    ///
    /// # Safety
    /// The library must be a trusted build artifact supplied by the
    /// operator; it is loaded into the process.
    pub unsafe fn load(checkpoint: &Path, library: &Path) -> Result<Self> {
        let (structure, _meta) = read_structure(checkpoint)?;
        profile().admit(&structure)?;
        let limits = &profile().limits;
        let cuda = unsafe { Cuda::load(library)? };
        let source = Weights::open(&checkpoint.join("model.safetensors"))?;
        source.validate(&structure)?;
        let mut weights = HashMap::new();
        for (name, shape) in omni_modernbert::geometry::encoder_tensors(&structure) {
            let values = source.f32(&name, &shape)?;
            let data = match crate::weights::storage_dtype(&name) {
                "f32" => bytes32(&values),
                "f16" => bytes16(
                    &values
                        .iter()
                        .map(|v| half::f16::from_f32(*v).to_bits())
                        .collect::<Vec<_>>(),
                ),
                _ => bytes16(
                    &values
                        .iter()
                        .map(|v| bf16::from_f32(*v).to_bits())
                        .collect::<Vec<_>>(),
                ),
            };
            let buffer = cuda
                .upload(&data)
                .with_context(|| format!("upload {name}"))?;
            weights.insert(name, buffer);
        }
        for width in [structure.hidden_size, 3 * structure.hidden_size] {
            let name = format!("zeros.{width}");
            weights.insert(name.clone(), cuda.upload(&vec![0; width * 4])?);
        }
        // Both attention kinds share one theta and one yarn scaling, so
        // one table set serves the plan's per-kind names.
        let len = limits.len_max.min(structure.max_positions);
        let (cos, sin, _tables) = cos_sin_tables(&structure, len)?;
        for kind in ["full", "local"] {
            weights.insert(format!("rope_{kind}_cos"), cuda.upload(&bytes32(&cos))?);
            weights.insert(format!("rope_{kind}_sin"), cuda.upload(&bytes32(&sin))?);
        }
        let mut plan = Vec::new();
        for step in plan::encoder_plan(&structure, &EncoderPlanOptions::default()) {
            let PlanStep::Launch { kernel, args } = step else {
                continue;
            };
            ensure!(args.len() <= 5, "encoder argument count");
            let mut resolved = [Arg::Weight(std::ptr::null_mut()); 5];
            for (slot, arg) in resolved.iter_mut().zip(&args) {
                *slot = match arg {
                    PlanArg::Weight(name) => {
                        Arg::Weight(weights[name].ptr())
                    }
                    PlanArg::Scratch(s) => Arg::Scratch(*s),
                };
            }
            plan.push(Step {
                kernel: cuda.resolve(symbol(&kernel))?,
                args: resolved,
                count: args.len(),
            });
        }
        Ok(Self {
            structure,
            cuda,
            _device_tensors: weights,
            plan,
        })
    }

    /// Run one sequence and return the final hidden at the requested
    /// positions, decoded from the bundle's bf16 working precision.
    pub fn encode(&self, ids: &[i64], positions: &[usize]) -> Result<Vec<Vec<f32>>> {
        let limits = &profile().limits;
        ensure!(!ids.is_empty(), "empty sequence");
        let len = ids.len().next_multiple_of(limits.len_step);
        limits.check_request(1, len)?;
        ensure!(
            positions.iter().all(|&p| p < ids.len()),
            "position outside the sequence"
        );
        let scratch = Scratch::new(&self.cuda, &self.structure, 1, len)?;
        let mut staging = vec![0i64; len];
        staging[..ids.len()].copy_from_slice(ids);
        scratch.ids.write(
            &staging.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
        )?;
        scratch
            .lens
            .write(&vec![ids.len() as i32; 1][..].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?;
        for step in &self.plan {
            let pointers: Vec<Ptr> = step.args[..step.count]
                .iter()
                .map(|arg| match arg {
                    Arg::Weight(p) => *p,
                    Arg::Scratch(slot) => scratch.buffer(*slot).ptr(),
                })
                .collect();
            // SAFETY: buffers are sized by the family geometry for this
            // admitted structure and stay alive through the sync below.
            unsafe { step.kernel.launch(&pointers, 1, len)? };
        }
        self.cuda.sync()?;
        let hidden = scratch.y.read(len * self.structure.hidden_size * 2)?;
        let decoded: Vec<f32> = hidden
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
            .collect();
        let width = self.structure.hidden_size;
        Ok(positions
            .iter()
            .map(|&p| decoded[p * width..(p + 1) * width].to_vec())
            .collect())
    }
}
