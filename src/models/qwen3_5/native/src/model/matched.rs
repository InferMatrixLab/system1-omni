//! Request-scoped selection of ordinary-compatible full-shape GEMM plans.
use super::*;

impl Model {
    /// Run aligned prefix windows with the first full prompt's GEMM algorithms.
    /// Returns `None` without invoking `execute` if independent full prompts use
    /// different descriptors or a window is unsupported. Legacy fixed-M64 calls
    /// outside this scope keep their own handle, including after errors.
    ///
    /// Descriptor equality is an admission check, not a portable cuBLAS numerical
    /// guarantee: requalify row invariance on each GPU/library/model combination.
    pub fn with_matched_prefix<T>(
        &mut self,
        lengths: &[usize],
        windows: &[usize],
        execute: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<Option<T>> {
        self.ensure_fixed_profile()?;
        ensure!(self.matched_ref == 0, "nested matched execution");
        ensure!(
            lengths
                .iter()
                .chain(windows)
                .all(|&m| m > 0 && m <= self.cfg.max_positions && m <= i32::MAX as usize),
            "invalid matched prompt/window length"
        );
        if lengths.len() < 2 || windows.is_empty() {
            return Ok(None);
        }
        let m = lengths[0];
        if !self.matched_handles.contains_key(&m) && self.matched_handles.len() == 16 {
            self.synchronize()?;
            // SAFETY: all launches drained before destroying owned workspaces.
            for (_, h) in self.matched_handles.drain() {
                unsafe { (cuda::api().cs1_gemm_destroy)(h) };
            }
        }
        let compatible = cuda::api()
            .matched_compatible()
            .context("rebuild libqwen3_5_cuda for matched prefix execution")?;
        let handle = if let Some(&h) = self.matched_handles.get(&m) {
            h
        } else {
            let create = cuda::api()
                .matched_create()
                .context("rebuild libqwen3_5_cuda for matched prefix execution")?;
            // SAFETY: validated positive i32 length; this model owns the handle.
            let h = unsafe { create(GEMM_WORKSPACE, m as i32) };
            ensure!(!h.is_null(), "matched cuBLASLt setup failed");
            self.matched_handles.insert(m, h);
            h
        };
        let mut checked = std::collections::HashSet::new();
        for layer in &self.layers {
            let mixer = match &layer.mixer {
                Mixer::Linear(l) => [&l.in_proj, &l.out],
                Mixer::Full(l) => [&l.qkv, &l.o],
            };
            for w in mixer.into_iter().chain([&layer.gate_up, &layer.down]) {
                ensure!(
                    w.shape.iter().all(|&n| n <= i32::MAX as usize),
                    "matched weight exceeds CUDA layout"
                );
                for (rows, ordinary) in lengths
                    .iter()
                    .map(|&n| (n, true))
                    .chain(windows.iter().map(|&n| (n, false)))
                {
                    if !checked.insert((rows, w.shape[0], w.shape[1], ordinary)) {
                        continue;
                    }
                    // SAFETY: owned handles and checked dimensions. Null ordinary
                    // asks only whether a prefix/suffix window is supported.
                    let okay = unsafe {
                        compatible(
                            if ordinary {
                                self.gemm
                            } else {
                                std::ptr::null_mut()
                            },
                            handle,
                            rows as i32,
                            w.shape[0] as i32,
                            w.shape[1] as i32,
                        )
                    };
                    if okay != 1 {
                        return Ok(None);
                    }
                }
            }
        }
        self.matched_ref = m;
        // The worker catches panics and can retain this engine. Restore the
        // ordinary selection and drain submitted work on every exit path.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| execute(self)));
        let synced = self.synchronize();
        self.matched_ref = 0;
        match result {
            Ok(value) => {
                synced?;
                value.map(Some)
            }
            Err(panic) => std::panic::resume_unwind(panic),
        }
    }
}
