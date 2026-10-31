use super::*;

#[test]
fn lru_keeps_recent_keys_within_the_budget() {
    let mut lru = Lru::new(10);
    assert_eq!(lru.admit("a", 4), Some(vec![]));
    assert_eq!(lru.admit("b", 4), Some(vec![]));
    assert_eq!((lru.bytes(), lru.len()), (8, 2));
    // a use keeps "a" past the next eviction
    assert!(lru.touch(&"a"));
    assert!(!lru.touch(&"c"));
    assert_eq!(lru.admit("c", 4), Some(vec!["b"]));
    assert_eq!((lru.bytes(), lru.len()), (8, 2));
    // a larger key drops as many old ones as it needs, oldest first
    assert!(lru.touch(&"c"));
    assert_eq!(lru.admit("d", 9), Some(vec!["a", "c"]));
    assert_eq!((lru.bytes(), lru.len()), (9, 1));
    // over the whole budget: refused, and nothing is dropped
    assert_eq!(lru.admit("e", 11), None);
    assert_eq!((lru.bytes(), lru.len()), (9, 1));
    lru.remove(&"d");
    assert!(lru.is_empty() && lru.bytes() == 0);
    lru.remove(&"d");
    assert_eq!(lru.bytes(), 0);
}

#[test]
fn stats_report_the_budget_before_any_use() {
    let cache = PrefixCache::new(1 << 30);
    assert_eq!(
        cache.stats(),
        PrefixCacheStats {
            budget_bytes: 1 << 30,
            ..Default::default()
        }
    );
}
