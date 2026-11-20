# Vela 2.0 0.3B native profile

`omni-vela-encoder-native` registers the Vela 2.0 0.3B encoder as the second
ModernBERT family profile and owns its CPU-verifiable half: bundle
config parsing (`config`), the registered profile and its bundle
limits (`profile`), the YaRN-scaled rotary tables (`rope`), and the
model-local checkpoint inventory with strict f32 accounting (`weights`).

Weight upload and encoder execution live behind the crate's `cuda`
feature (`executor`): the admitted structure runs as the family core
plan against the regenerated shared bundle, whose ABI-2 kernels serve
both family profiles — the Laya 1024-hidden geometry under its
unchanged symbols and this profile's 768-hidden geometry under
`_h768`-suffixed names. Everything CPU-side is validated against the
pinned checkpoint's own files; see the tests under
[`tests/vela/native/`](../../../../tests/vela/native/) and the pinned
census and rotary-reference fixtures they consume.

The serving path today is the Python reference worker documented in
[`src/models/vela`](../README.md) and
[`recipe/vela/reference.md`](../../../../recipe/vela/reference.md);
its frozen parity fixtures are the comparison baseline this native
route will be judged against.

## Workspace coexistence and checks

The decoder members (0.8B/4B/9B) keep the separate
`omni-vela-native` package under [`../native/`](../native/); their
processing, heads and Qwen executor are documented in the
[decoder recipe](../../../../recipe/vela/native.md). This encoder
carrier reuses the single [`omni-modernbert`](../../modernbert/README.md)
family core alongside Laya. Moving the carrier changes package and
import paths only; the profile, weights, YaRN arithmetic and fixtures
retain their pinned contract.

From the repository root, discover and run the ordinary CPU suites:

```sh
cargo test --locked -p omni-vela-encoder-native -p omni-vela-native -- --list
cargo test --locked -p omni-vela-encoder-native -p omni-vela-native
cargo test --locked -p omni-vela-encoder-native --features cuda -- --list
```

The existing encoder executor opt-in is now selected with
`cargo test --locked -p omni-vela-encoder-native --features cuda --test vela_executor`.
It requires `VELA_NATIVE_CHECKPOINT` and `VELA_NATIVE_LIBRARY` and
returns early when either is missing; an ordinary CPU run does not
establish device execution. The inherited bundle and build gate remain
Hopper-only (`sm_90a`); L20 (`sm_89`) execution is unsupported by this
encoder backend. No native encoder HTTP worker is provided here.
