//! Exact real-weight stage regression; external oracle generated in the pinned env.
use super::*;

#[test]
#[ignore = "needs GPU, OMNIJEV_EXPORT, CUA_S1_CUDA_LIB and OMNIJEV_VISION_ORACLE"]
fn omnijev_reference_matches_official_vision_stages_and_graph_replay() {
    let path = |name| std::path::PathBuf::from(std::env::var_os(name).expect(name));
    let oracle = path("OMNIJEV_VISION_ORACLE");
    let fixture: serde_json::Value =
        serde_json::from_slice(&std::fs::read(oracle.join("reference.json")).unwrap()).unwrap();
    assert_eq!(fixture["schema"], "omnijev-4b-vision-reference/1");
    assert_eq!(fixture["mso1_output_exact"], true);
    let grid: [usize; 3] = serde_json::from_value(fixture["grid"].clone()).unwrap();
    let bytes = std::fs::read(oracle.join("pixels.f32")).unwrap();
    assert_eq!(bytes.len(), grid[1] * grid[2] * 1536 * 4);
    let image = ProcessedImage {
        pixel_values: bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&b| f32::from_le_bytes(b))
            .collect(),
        image_grid_thw: grid,
        resized_height: grid[1] * 16,
        resized_width: grid[2] * 16,
    };
    let mut model = VisionModel::load(path("OMNIJEV_EXPORT"), &path("CUA_S1_CUDA_LIB")).unwrap();
    assert!(
        model.enable_reference_numerics().is_err(),
        "4B must not silently select JEMM"
    );
    model.cache_limits = Some(CacheLimits {
        entries: 1,
        bytes: usize::MAX,
    });
    assert!(
        model
            .enable_omnijev_numerics()
            .unwrap_err()
            .to_string()
            .contains("bounded cache")
    );
    model.cache_limits = None;
    model
        .lora
        .insert("rejected_adapter".into(), DeviceBuffer::new(0).unwrap());
    assert!(
        model
            .enable_omnijev_numerics()
            .unwrap_err()
            .to_string()
            .contains("unadapted 4B")
    );
    model.lora.clear();
    model.enable_omnijev_numerics().unwrap();
    model.enable_omnijev_numerics().unwrap();
    let expected: BTreeMap<String, Vec<u8>> = fixture["stages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|stage| {
            let name = stage["name"].as_str().unwrap().to_owned();
            let bytes = std::fs::read(oracle.join(format!("{name}.bf16"))).unwrap();
            let shape: Vec<usize> = serde_json::from_value(stage["shape"].clone()).unwrap();
            assert_eq!(shape.iter().product::<usize>() * 2, bytes.len(), "{name}");
            (name, bytes)
        })
        .collect();
    let mut seen = std::collections::BTreeSet::new();
    let result = model
        .forward_with_block_trace(&image, 0, |name, values| {
            let actual: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            let wanted = &expected[name];
            let different = actual
                .as_chunks::<2>()
                .0
                .iter()
                .zip(wanted.as_chunks::<2>().0.iter())
                .filter(|(a, b)| a != b)
                .count();
            assert_eq!(different, 0, "{name}: BF16 values differ");
            assert_eq!(actual.len(), wanted.len(), "{name}");
            seen.insert(name.to_owned());
            Ok(())
        })
        .unwrap();
    assert_eq!(seen, expected.keys().cloned().collect());
    assert!(
        model
            .enable_omnijev_numerics()
            .unwrap_err()
            .to_string()
            .contains("before the first forward")
    );
    let bits = |values: &[bf16]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    model.graph_enabled = true;
    assert_eq!(bits(&model.forward(&image).unwrap()), bits(&result));
    assert!(
        model.scratch[0].graph.is_some(),
        "capture must actually succeed"
    );
    assert_eq!(bits(&model.forward(&image).unwrap()), bits(&result));
    assert_eq!(
        bits(&model.forward_device(&image).unwrap().download().unwrap()),
        bits(&result)
    );
    let mut changed = ProcessedImage {
        pixel_values: image.pixel_values.clone(),
        image_grid_thw: grid,
        resized_height: image.resized_height,
        resized_width: image.resized_width,
    };
    for pixel in changed.pixel_values.iter_mut().take(1536) {
        *pixel += 0.125;
    }
    model.graph_enabled = false;
    let changed_eager = model.forward(&changed).unwrap();
    assert_ne!(bits(&changed_eager), bits(&result));
    model.graph_enabled = true;
    assert_eq!(
        bits(&model.forward(&changed).unwrap()),
        bits(&changed_eager)
    );
    assert_eq!(bits(&model.forward(&image).unwrap()), bits(&result));
    assert!(
        model
            .forward_with_block_trace(&image, model.config.depth, |_, _| Ok(()))
            .is_err()
    );
    assert!(
        model
            .forward_with_block_trace(&image, 0, |_, _| anyhow::bail!("trace error"))
            .is_err()
    );
    assert_eq!(bits(&model.forward(&image).unwrap()), bits(&result));
}
