//! Window Graphs retain every prefix allocation whose address they record.
use super::*;

pub(super) struct Entry {
    key: [usize; 5],
    // Destroy the Graph before releasing recorded allocations.
    graph: cuda::Graph,
    _states: [Option<Arc<PrefixBuffers>>; 2],
}

impl Model {
    pub(super) fn run_window_graph(
        &mut self,
        begin: usize,
        end: usize,
        mut capture: Option<&mut PrefixState>,
        prefix: Option<&PrefixState>,
        gemm: *mut c_void,
    ) -> Result<()> {
        let s = self.scratch.as_ref().unwrap();
        if !self.graph_enabled {
            return self.run_window(s, begin, end, true, capture, prefix, gemm);
        }
        let states = [
            prefix.map(|p| Arc::clone(&p.buffers)),
            capture.as_deref().map(|p| Arc::clone(&p.buffers)),
        ];
        let identities = states
            .each_ref()
            .map(|p| p.as_ref().map_or(0, |p| Arc::as_ptr(p) as usize));
        let key = [begin, end, identities[0], identities[1], gemm as usize];
        if let Some(entry) = self.window_graphs.iter().find(|entry| entry.key == key) {
            entry.graph.launch(self.stream)?;
            self.graph_stats.replays += 1;
            if std::env::var("CUA_S1_GRAPH_TRACE").as_deref() == Ok("1") {
                eprintln!("CUA_S1_GRAPH mode=window event=replay begin={begin} end={end}");
            }
            return Ok(());
        }
        // Preserve the eager miss result; capture records without executing.
        self.run_window(s, begin, end, true, capture.as_deref_mut(), prefix, gemm)?;
        cuda::synchronize(self.stream)?;
        let recorded = cuda::Graph::capture(self.stream, || {
            self.run_window(s, begin, end, true, capture, prefix, gemm)
        });
        self.cache_window_graph(key, states, recorded)
    }

    pub(super) fn cache_window_graph(
        &mut self,
        key: [usize; 5],
        states: [Option<Arc<PrefixBuffers>>; 2],
        recorded: Result<cuda::Graph>,
    ) -> Result<()> {
        let [begin, end, ..] = key;
        match recorded {
            Ok(graph) => {
                if self.window_graphs.len() == 128 {
                    cuda::synchronize(self.stream)?;
                    self.window_graphs.pop_front();
                }
                self.window_graphs.push_back(Entry {
                    key,
                    graph,
                    _states: states,
                });
                self.graph_stats.captures += 1;
                if std::env::var("CUA_S1_GRAPH_TRACE").as_deref() == Ok("1") {
                    eprintln!("CUA_S1_GRAPH mode=window event=capture begin={begin} end={end}");
                }
            }
            Err(error) => {
                // Existing retirement path drains, disables all language Graph
                // modes and retains the already computed eager miss result.
                self.cache_graph(&[begin, end], true, Err(error))?;
            }
        }
        Ok(())
    }
}
