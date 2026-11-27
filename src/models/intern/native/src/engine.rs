use std::{
    path::Path,
    sync::{Arc, Mutex},
    time::Instant,
};

use anyhow::{Result, anyhow};
use omni_runtime::SerialScheduler;
use serde_json::Value;

use crate::{executor::Executor, finishing, processing::Processor};

pub struct Engine {
    processor: Processor,
    scheduler: SerialScheduler,
    executor: Arc<Mutex<Executor>>,
}

impl Engine {
    pub async fn load(checkpoint: &Path, library: &Path) -> Result<Self> {
        let checkpoint = checkpoint.to_path_buf();
        let library = library.to_path_buf();
        let (processor, executor) = tokio::task::spawn_blocking(move || {
            let processor = Processor::load(&checkpoint.join("tokenizer.json"))?;
            let executor = Executor::load(&checkpoint, &library)?;
            Ok::<_, anyhow::Error>((processor, executor))
        })
        .await??;
        Ok(Self {
            processor,
            executor: Arc::new(Mutex::new(executor)),
            scheduler: SerialScheduler::default(),
        })
    }

    pub async fn decide(&self, body: &[u8]) -> Result<Value> {
        let prepared = self.processor.prepare(body)?;
        let symbols = self.processor.symbol_ids.clone();
        let executor = Arc::clone(&self.executor);
        self.scheduler
            .run(move || {
                let mut executor = executor
                    .lock()
                    .map_err(|_| anyhow!("Intern executor is poisoned"))?;
                let start = Instant::now();
                let hidden = executor.hidden(&prepared.ids, &prepared.readout)?;
                let logits = executor.project(&hidden, &symbols)?;
                let inference_ms = start.elapsed().as_secs_f64() * 1000.0;
                finishing::finish(&prepared.fields, &logits, prepared.ids.len(), inference_ms)
            })
            .await
    }
}
