//! Fixture loading shared by the OmniJev tests.
#![allow(dead_code)]

use std::io::Cursor;

use base64::Engine;
use serde_json::Value;

pub fn fixture(name: &str) -> Value {
    let path = format!(
        "{}/../../../../tests/omnijev/data/{name}",
        env!("CARGO_MANIFEST_DIR")
    );
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// A solid-colour PNG as a data URL; every encoder gives these pixels.
pub fn png_url(width: u32, height: u32, colour: [u8; 3]) -> String {
    let image = image::RgbImage::from_pixel(width, height, image::Rgb(colour));
    let mut bytes = Vec::new();
    image
        .write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)
        .unwrap();
    format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(bytes)
    )
}

/// A processing fixture's request, with its image generated.
pub fn request(case: &Value) -> Vec<u8> {
    let mut request = case["request"].clone();
    let size = &case["image_size"];
    let colour = &case["image_colour"];
    let rgb = [0, 1, 2].map(|i| colour[i].as_u64().unwrap() as u8);
    request["state"]["images"][0] = Value::String(png_url(
        size[0].as_u64().unwrap() as u32,
        size[1].as_u64().unwrap() as u32,
        rgb,
    ));
    serde_json::to_vec(&request).unwrap()
}

/// Fixture token ids, with the image run (stored as its negated length) expanded.
pub fn token_ids(collapsed: &Value) -> Vec<u32> {
    let mut ids = Vec::new();
    for v in collapsed.as_array().unwrap() {
        let v = v.as_i64().unwrap();
        if v < 0 {
            ids.extend(std::iter::repeat_n(248_056, (-v) as usize));
        } else {
            ids.push(v as u32);
        }
    }
    ids
}
