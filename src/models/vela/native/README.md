# Vela 2.0 0.3B native profile

`omni-vela-native` registers the Vela 2.0 0.3B encoder as the second
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
