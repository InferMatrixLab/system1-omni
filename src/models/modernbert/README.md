# ModernBERT encoder family core

`omni-modernbert` owns the family data shared by the ModernBERT-based
encoders: structure parameters parsed from checkpoint configs with
per-field provenance ([`structure`](src/structure.rs)), the tensor
geometry derived from a structure ([`geometry`](src/geometry.rs)),
registered capability profiles that decide whether a CUDA bundle may
execute a structure ([`gate`](src/gate.rs)), and the named-kernel
execution plan for the encoder stack ([`plan`](src/plan.rs)).

The crate is family data only. It never touches CUDA, device memory, or
weight files: loading weights, resolving kernel symbols to device
pointers, and every task head stay with the model crates that consume a
profile. The Laya native worker runs its encoder on this core
([`src/models/laya`](../laya/README.md)), and the Vela 2.0 0.3B native
profile is the second registered family member
([`src/models/vela/native`](../vela/native/README.md)).

Structure parsing and capability admission are CPU-verifiable without a
checkpoint or device: the tests under
[`tests/modernbert/`](../../../tests/modernbert/) pin adapter
provenance, family invariants, the frozen gate's accept/reject set, and
the serialized execution plans as golden fixtures.
