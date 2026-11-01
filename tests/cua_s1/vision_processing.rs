use super::*;

#[test]
fn resize_workers_are_explicit_bounded_and_default_off() {
    assert_eq!(resize_workers(None).unwrap(), 0);
    for value in 0..=4 {
        assert_eq!(resize_workers(Some(&value.to_string())).unwrap(), value);
    }
    for value in ["", "5", "true", "+1", "01", " 1"] {
        assert!(resize_workers(Some(value)).is_err());
    }
}
