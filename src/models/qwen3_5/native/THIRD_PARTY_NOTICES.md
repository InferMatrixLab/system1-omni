# Image preprocessing attributions

`src/image_preprocess.rs` is a Rust adaptation of the following algorithms.
It is modified for decoded interleaved RGB8 input, validated Qwen3.5/3.8 vision settings,
bounded dimensions, standard-library buffers, and a standalone CPU API.

- PyTorch 2.14.0, `aten/src/ATen/native/cpu/UpSampleKernel.cpp`
  (`_compute_indices_min_size_weights_aa`, `_compute_index_ranges_int16_weights`,
  and the separable uint8 horizontal/vertical loops), plus the cubic polynomial
  helpers in `aten/src/ATen/native/UpSample.h`. See [PyTorch license](licenses/PYTORCH-LICENSE)
  for the retained copyright notices, redistribution conditions, and disclaimer.
- PyTorch's bicubic filter credits Pillow's `src/libImaging/Resample.c`.
  The retained PIL/Pillow notice is in [Pillow license](licenses/PILLOW-LICENSE).
- Transformers 5.17.0,
  `src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py` (smart resize
  and patch ordering) and `src/transformers/image_processing_backends.py`
  (fused normalization). Copyright 2024 The Qwen team, Alibaba Group and the
  HuggingFace Inc. team. All rights reserved. The backend file is
  Copyright 2025 The HuggingFace Inc. team. Licensed under the
  [Apache License, Version 2.0](licenses/APACHE-2.0).

No upstream runtime or image decoder is linked by this module. The above
notices and license texts must accompany redistributed adaptations as required
by their respective licenses.

The native vision geometry, rotary, block and merger execution in
`src/vision/` follows Transformers 5.17.0 `modeling_qwen3_5.py`, Copyright
2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team, under the
[Apache License, Version 2.0](licenses/APACHE-2.0). The implementation is
adapted to Rust/CUDA optional separate FP32 visual LoRA execution and the shared 4B/27B layouts.

## Kev default-service attention

`src/backends/cuda/qwen3_5/service_readout.cu` instantiates the memory-efficient
attention templates from PyTorch 2.8.0
`ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h`, Copyright
(c) Meta Platforms, Inc. and affiliates, under the retained
[PyTorch 2.8 license](licenses/PYTORCH-2.8-LICENSE). Its CUTLASS dependency is
pinned to `ad7b2f5e84fcfa124cb02b91d5bd26d238c0459e`, Copyright (c) 2017–2025
NVIDIA CORPORATION & AFFILIATES, under the retained
[CUTLASS license](licenses/CUTLASS-LICENSE).

The templates are compiled into the optional CUDA library; no Python or libtorch
runtime is linked. `recipe/kev/build_cuda.sh` places both license texts and
`KEV_CUDA_NOTICES.txt` beside the resulting library. Include these files when
redistributing that binary. The ordinary shared Qwen build does not instantiate
this optional attention kernel.
