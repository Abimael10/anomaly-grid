//! Concurrency invariants for anomaly-grid.
//!
//! `batch_score` parallelises detection via rayon. The detector is
//! shared across worker threads as `&AnomalyDetector` — these tests
//! pin down the static guarantees that make that sound.

use anomaly_grid::{batch_score, AnomalyDetector, AnomalyGridConfig, AnomalyScore, ContextTree, MarkovModel};
use static_assertions::assert_impl_all;

assert_impl_all!(AnomalyDetector: Send, Sync);
assert_impl_all!(MarkovModel: Send, Sync);
assert_impl_all!(ContextTree: Send, Sync);
assert_impl_all!(AnomalyScore: Send, Sync);
assert_impl_all!(AnomalyGridConfig: Send, Sync);

/// Same input must produce the same output across rayon thread-pool sizes.
/// Catches non-determinism from iteration order or false sharing.
#[test]
fn batch_score_is_deterministic_across_thread_counts() {
    let mut detector = AnomalyDetector::new(3).expect("detector");
    let normal: Vec<String> = "ABCABCABCABCABCABC"
        .chars()
        .map(|c| c.to_string())
        .collect();
    detector.train(&normal).expect("train");

    let test_seqs: Vec<Vec<String>> = (0..50)
        .map(|i| {
            let mut seq: Vec<String> = "ABCABC".chars().map(|c| c.to_string()).collect();
            // Inject one rare token at a varying position to give each
            // sequence a slightly different score profile.
            let len = seq.len();
            seq[i % len] = "X".to_string();
            seq
        })
        .collect();

    let baseline = run_with_threads(&detector, &test_seqs, 1);

    for threads in [2, 4, 8] {
        let parallel = run_with_threads(&detector, &test_seqs, threads);
        assert_eq!(
            baseline.len(),
            parallel.len(),
            "result length differs at {threads} threads"
        );
        for (i, (a, b)) in baseline.iter().zip(parallel.iter()).enumerate() {
            assert_eq!(
                a.len(),
                b.len(),
                "score-list length differs for sequence {i} at {threads} threads"
            );
            for (sa, sb) in a.iter().zip(b.iter()) {
                assert_eq!(sa.sequence, sb.sequence);
                assert!(
                    (sa.likelihood - sb.likelihood).abs() < 1e-12,
                    "likelihood diverged at {threads} threads"
                );
                assert!(
                    (sa.anomaly_strength - sb.anomaly_strength).abs() < 1e-12,
                    "anomaly_strength diverged at {threads} threads"
                );
                assert!(
                    (sa.information_score - sb.information_score).abs() < 1e-12,
                    "information_score diverged at {threads} threads"
                );
            }
        }
    }
}

fn run_with_threads(
    detector: &AnomalyDetector,
    seqs: &[Vec<String>],
    threads: usize,
) -> Vec<Vec<AnomalyScore>> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("rayon pool");
    pool.install(|| batch_score(detector, seqs, 0.0).expect("batch_score"))
}

/// Parallel `batch_score` must equal a sequential loop calling
/// `detect_anomalies` on the same detector — i.e. parallelism is purely
/// a performance concern, not a semantic one.
#[test]
fn batch_score_matches_sequential_loop() {
    let mut detector = AnomalyDetector::new(2).expect("detector");
    detector
        .train(&["A", "B", "C", "A", "B", "C", "A", "B", "C", "A"]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>())
        .expect("train");

    let test_seqs: Vec<Vec<String>> = (0..30)
        .map(|i| {
            let mut s: Vec<String> = "ABCABC".chars().map(|c| c.to_string()).collect();
            if i % 7 == 0 {
                s.push("Z".to_string());
            }
            s
        })
        .collect();

    let parallel = batch_score(&detector, &test_seqs, 0.0).expect("parallel");
    let sequential: Vec<Vec<AnomalyScore>> = test_seqs
        .iter()
        .map(|s| detector.detect_anomalies(s, 0.0).expect("detect"))
        .collect();

    assert_eq!(parallel.len(), sequential.len());
    for (i, (par, seq)) in parallel.iter().zip(sequential.iter()).enumerate() {
        assert_eq!(par.len(), seq.len(), "size mismatch at sequence {i}");
        for (a, b) in par.iter().zip(seq.iter()) {
            assert_eq!(a.sequence, b.sequence);
            assert!((a.anomaly_strength - b.anomaly_strength).abs() < 1e-12);
        }
    }
}
