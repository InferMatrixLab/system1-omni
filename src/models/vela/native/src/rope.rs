use anyhow::{Context, Result, ensure};
use omni_modernbert::structure::{ModernBertStructure, RopeSpec, YarnScaling};

/// YaRN-derived rotary parameters: the interpolated inverse frequencies
/// and the attention scaling the reference applies beside them. Both
/// attention kinds share one set (the pinned config states one theta
/// and one rope_scaling).
#[derive(Debug, Clone, PartialEq)]
pub struct YarnTables {
    /// `rotary_half` entries; row `p` of a cos/sin table is
    /// `p * inv_freq[j]` passed through cos/sin.
    pub inv_freq: Vec<f32>,
    /// `1 + 0.1 * ln(factor)` in the reference's default-mscale path.
    pub attention_scaling: f64,
}

/// Computes the YaRN inverse frequencies exactly as the reference
/// runtime does (transformers' yarn initialization): extrapolation and
/// interpolation frequencies blended by a linear ramp whose bounds come
/// from the correction-dimension formula, floored/ceiled under
/// `truncate`, and the default attention scaling. Internals run in
/// f64, outputs narrow to f32.
pub fn yarn(rope: &RopeSpec, rotary_half: usize) -> Result<YarnTables> {
    let YarnScaling {
        factor,
        original_max_positions,
        beta_fast,
        beta_slow,
    } = rope.scaling.as_ref().context("rope has no yarn scaling")?;
    let base = rope.theta;
    let half = rotary_half as f64;
    let correction_dim = |rotations: f64| {
        (2.0 * half * (*original_max_positions as f64 / (rotations * std::f64::consts::TAU)).ln())
            / (2.0 * base.ln())
    };
    let low = correction_dim(*beta_fast).floor().max(0.0);
    // The reference clamps the ramp's upper bound to the head dimension
    // (2 * rotary_half) - 1, not to the ramp array's last index: with
    // `truncate`, a bound above the array saturates the mask instead of
    // rescaling the in-band ramp. Clamping to rotary_half - 1 would
    // shrink (high - low) and change every in-band blend.
    let high = correction_dim(*beta_slow)
        .ceil()
        .min(2.0 * rotary_half as f64 - 1.0);
    ensure!(high > low, "degenerate yarn ramp bounds {low}..{high}");
    let ramp = |j: f64| ((j - low) / (high - low)).clamp(0.0, 1.0);
    let inv_freq = (0..rotary_half)
        .map(|j| {
            let j = j as f64;
            let pos_freq = base.powf(j / half);
            let interpolation = 1.0 / (factor * pos_freq);
            let extrapolation = 1.0 / pos_freq;
            let ramp = ramp(j);
            (interpolation * ramp + extrapolation * (1.0 - ramp)) as f32
        })
        .collect();
    Ok(YarnTables {
        inv_freq,
        attention_scaling: 1.0 + 0.1 * factor.ln(),
    })
}

/// Full cos/sin tables for positions `[0, len)` in f32, row-major
/// `[len, rotary_half]` each — the layout the family rope kernel
/// consumes. The reference runtime multiplies the YaRN attention
/// scaling into its tables, so these do too; the kernel's own
/// attention scale stays unchanged and must not apply it twice.
pub fn cos_sin_tables(
    structure: &ModernBertStructure,
    len: usize,
) -> Result<(Vec<f32>, Vec<f32>, YarnTables)> {
    ensure!(
        len > 0 && len <= structure.max_positions,
        "table length {len}"
    );
    let rope = &structure.rope_full;
    ensure!(
        structure.rope_full == structure.rope_sliding,
        "the two attention kinds must share one rope"
    );
    let tables = yarn(rope, omni_modernbert::geometry::rotary_half(structure))?;
    let half = tables.inv_freq.len();
    // Fail closed on a corrupt structure instead of attempting an
    // oversized allocation: `max_positions` is untrusted input until a
    // profile admits it.
    let elements = len
        .checked_mul(half)
        .with_context(|| format!("table length {len} overflows the table size"))?;
    let mut cos = Vec::new();
    let mut sin = Vec::new();
    cos.try_reserve(elements)
        .with_context(|| format!("cannot allocate a {len}x{half} cosine table"))?;
    sin.try_reserve(elements)
        .with_context(|| format!("cannot allocate a {len}x{half} sine table"))?;
    let scaling = tables.attention_scaling as f32;
    for pos in 0..len {
        for &f in &tables.inv_freq {
            let angle = pos as f32 * f;
            cos.push(angle.cos() * scaling);
            sin.push(angle.sin() * scaling);
        }
    }
    Ok((cos, sin, tables))
}
