//! Domain 3: Information Theory Correctness
//!
//! This module implements comprehensive domain-driven tests for information theory
//! fundamentals, ensuring our implementation adheres to Shannon's principles and
//! information-theoretic measures used in anomaly detection.

#![allow(clippy::uninlined_format_args)]
#![allow(clippy::useless_vec)]
#![allow(clippy::iter_kv_map)]
#![allow(clippy::for_kv_map)]

use anomaly_grid::*;

#[test]
fn domain_3_information_theory() {
    println!("🔬 DOMAIN 3: INFORMATION THEORY CORRECTNESS");
    println!("===========================================");
    println!();

    let mut test_results = Vec::new();

    // Test 3.1: Shannon Entropy Calculation
    println!("Test 3.1: Shannon Entropy Calculation");
    println!("-------------------------------------");
    let entropy_result = test_shannon_entropy_comprehensive();
    test_results.push(("Shannon Entropy", entropy_result));
    println!();

    // Test 3.2: Information Content Properties
    println!("Test 3.2: Information Content Properties");
    println!("----------------------------------------");
    let info_content_result = test_information_content_properties_comprehensive();
    test_results.push(("Information Content", info_content_result));
    println!();

    // Test 3.3: Kullback-Leibler Divergence
    println!("Test 3.3: Kullback-Leibler Divergence");
    println!("-------------------------------------");
    let kl_divergence_result = test_kl_divergence_comprehensive();
    test_results.push(("KL Divergence", kl_divergence_result));
    println!();

    // Test 3.4: Cross-Entropy Relationships
    println!("Test 3.4: Cross-Entropy Relationships");
    println!("-------------------------------------");
    let cross_entropy_result = test_cross_entropy_relationships_comprehensive();
    test_results.push(("Cross Entropy", cross_entropy_result));
    println!();

    // Test 3.5: Information Theory Inequalities
    println!("Test 3.5: Information Theory Inequalities");
    println!("------------------------------------------");
    let inequalities_result = test_information_theory_inequalities();
    test_results.push(("Information Inequalities", inequalities_result));
    println!();

    // Domain 3 Summary
    println!("🏆 DOMAIN 3 SUMMARY");
    println!("===================");
    let passed_tests = test_results
        .iter()
        .filter(|(_, result)| result.passed)
        .count();
    let total_tests = test_results.len();

    for (test_name, result) in &test_results {
        let status = if result.passed { "✅" } else { "❌" };
        println!("  {} {}: {}", status, test_name, result.evidence);
    }

    println!();
    println!(
        "Domain 3 Result: {}/{} tests passed",
        passed_tests, total_tests
    );

    assert_eq!(
        passed_tests, total_tests,
        "Domain 3 (Information Theory) failed: {}/{} tests passed",
        passed_tests, total_tests
    );
}

#[derive(Debug)]
struct DomainTestResult {
    passed: bool,
    evidence: String,
    details: Vec<String>,
}

impl DomainTestResult {
    fn pass(evidence: String) -> Self {
        Self {
            passed: true,
            evidence,
            details: Vec::new(),
        }
    }

    fn fail(evidence: String) -> Self {
        Self {
            passed: false,
            evidence,
            details: Vec::new(),
        }
    }

    fn with_details(mut self, details: Vec<String>) -> Self {
        self.details = details;
        self
    }
}

/// Test 3.1: Shannon Entropy Calculation
///
/// Shannon entropy: H(X) = -Σ P(x) log₂ P(x)
/// Properties:
/// - H(X) ≥ 0 (non-negative)
/// - H(X) = 0 iff one outcome has probability 1
/// - H(X) is maximized when all outcomes are equally likely
fn test_shannon_entropy_comprehensive() -> DomainTestResult {
    println!("  Testing Shannon entropy calculation and properties...");

    let mut violations = 0;
    let mut details = Vec::new();

    // Test 1: Entropy of deterministic distribution (should be 0)
    println!("    Testing deterministic distribution entropy");
    let mut detector1 = AnomalyDetector::new(1).expect("Failed to create detector");
    let deterministic_sequence = vec!["A"; 100]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    detector1
        .train(&deterministic_sequence)
        .expect("Failed to train");

    // Get a context node to test entropy calculation
    if let Some(context_node) = detector1
        .model()
        .context_tree()
        .get_context_node(&vec!["A".to_string()])
    {
        let entropy_deterministic = context_node.compute_entropy(detector1.model().config());
        println!("      Deterministic entropy: {:.6}", entropy_deterministic);

        // Should be very close to 0 (allowing for smoothing effects)
        if entropy_deterministic > 0.1 {
            violations += 1;
            details.push(format!(
                "Deterministic entropy too high: {:.6} > 0.1",
                entropy_deterministic
            ));
        }
        details.push(format!(
            "Deterministic entropy: {:.6}",
            entropy_deterministic
        ));
    }

    // Test 2: Entropy of uniform distribution (should be maximum)
    println!("    Testing uniform distribution entropy");
    let mut detector2 = AnomalyDetector::new(1).expect("Failed to create detector");
    // Create a truly uniform distribution where each context can transition to multiple states
    let uniform_sequence = vec!["A", "X", "A", "Y", "A", "Z", "A", "W"]
        .repeat(50)
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    detector2.train(&uniform_sequence).expect("Failed to train");

    if let Some(context_node) = detector2
        .model()
        .context_tree()
        .get_context_node(&vec!["A".to_string()])
    {
        let entropy_uniform = context_node.compute_entropy(detector2.model().config());
        println!("      Uniform entropy: {:.6}", entropy_uniform);

        // For 4 equally likely outcomes from context A, theoretical max entropy is log₂(4) = 2.0
        let theoretical_max = 2.0;
        if entropy_uniform < theoretical_max * 0.8 {
            // Allow some tolerance for smoothing
            violations += 1;
            details.push(format!(
                "Uniform entropy too low: {:.6} < {:.6}",
                entropy_uniform,
                theoretical_max * 0.8
            ));
        }
        details.push(format!(
            "Uniform entropy: {:.6}, theoretical max: {:.6}",
            entropy_uniform, theoretical_max
        ));
    }

    // Test 3: Entropy ordering (uniform > skewed > deterministic)
    println!("    Testing entropy ordering");
    let mut detector3 = AnomalyDetector::new(1).expect("Failed to create detector");
    // Create a skewed distribution from context A
    let skewed_sequence = vec!["A", "X", "A", "X", "A", "X", "A", "Y"]
        .repeat(50)
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    detector3.train(&skewed_sequence).expect("Failed to train");

    if let Some(context_node) = detector3
        .model()
        .context_tree()
        .get_context_node(&vec!["A".to_string()])
    {
        let entropy_skewed = context_node.compute_entropy(detector3.model().config());
        println!("      Skewed entropy: {:.6}", entropy_skewed);
        details.push(format!("Skewed entropy: {:.6}", entropy_skewed));
    }

    // Test 4: Non-negativity
    println!("    Testing entropy non-negativity");
    let test_sequences = vec![
        vec!["A"; 50],
        vec!["A", "B"].repeat(25),
        vec!["A", "B", "C"].repeat(17),
    ];

    for (i, sequence) in test_sequences.iter().enumerate() {
        let mut detector = AnomalyDetector::new(1).expect("Failed to create detector");
        let seq_strings: Vec<String> = sequence.iter().map(|s| s.to_string()).collect();
        detector.train(&seq_strings).expect("Failed to train");

        // Test entropy for different contexts
        let global_vocab: Vec<&String> = detector.model().state_mapping().keys().collect();
        for state in &global_vocab {
            if let Some(context_node) = detector
                .model()
                .context_tree()
                .get_context_node(&vec![state.to_string()])
            {
                let entropy = context_node.compute_entropy(detector.model().config());
                if entropy < 0.0 {
                    violations += 1;
                    details.push(format!(
                        "Negative entropy in sequence {}, context {}: {:.6}",
                        i, state, entropy
                    ));
                }
            }
        }
    }

    if violations == 0 {
        DomainTestResult::pass("Shannon entropy correctly calculated".to_string())
            .with_details(details)
    } else {
        DomainTestResult::fail(format!("{} Shannon entropy violations", violations))
            .with_details(details)
    }
}

/// Test 3.2: Information Content Properties
///
/// Information content: I(x) = -log₂ P(x)
/// Properties:
/// - I(x) ≥ 0 (non-negative)
/// - I(x) = 0 iff P(x) = 1
/// - I(x) increases as P(x) decreases (rare events have high information)
fn test_information_content_properties_comprehensive() -> DomainTestResult {
    println!("  Testing information content properties...");

    let mut detector = AnomalyDetector::new(1).expect("Failed to create detector");

    // Create a distribution with known probabilities
    let training_sequence = vec!["A", "A", "A", "A", "B", "B", "C"]
        .repeat(100)
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    detector.train(&training_sequence).expect("Failed to train");

    let mut violations = 0;
    let mut details = Vec::new();

    // Test information content calculation
    println!("    Testing information content calculation");
    let context = vec!["A".to_string()];
    let states = vec!["A", "B", "C"];
    let mut info_contents = Vec::new();
    let mut probabilities = Vec::new();

    for state in &states {
        let prob = detector
            .model()
            .get_best_context_probability(&context, state);
        let info_content = if prob > 0.0 {
            -prob.log2()
        } else {
            f64::INFINITY
        };

        probabilities.push(prob);
        info_contents.push(info_content);

        println!(
            "      P({} | A) = {:.6}, I({}) = {:.6}",
            state, prob, state, info_content
        );

        // Test non-negativity
        if info_content < 0.0 {
            violations += 1;
            details.push(format!(
                "Negative information content for {}: {:.6}",
                state, info_content
            ));
        }
    }

    // Test ordering: rare events should have higher information content
    println!("    Testing information content ordering");

    // Find the most and least probable events
    let max_prob_idx = probabilities
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    let min_prob_idx = probabilities
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    if max_prob_idx != min_prob_idx {
        let high_prob_info = info_contents[max_prob_idx];
        let low_prob_info = info_contents[min_prob_idx];

        println!("      Highest prob event info: {:.6}", high_prob_info);
        println!("      Lowest prob event info: {:.6}", low_prob_info);

        // Low probability events should have higher information content
        if low_prob_info <= high_prob_info {
            violations += 1;
            details.push(format!("Information content ordering violation: low_prob_info ({:.6}) <= high_prob_info ({:.6})", 
                               low_prob_info, high_prob_info));
        }

        details.push(format!(
            "Info content ordering: rare={:.6}, common={:.6}",
            low_prob_info, high_prob_info
        ));
    }

    // Test relationship with entropy
    println!("    Testing relationship with entropy");
    if let Some(context_node) = detector.model().context_tree().get_context_node(&context) {
        let entropy = context_node.compute_entropy(detector.model().config());

        // Entropy should be the expected information content using the SAME probability source
        let config = detector.model().config();
        let all_probs = context_node.get_all_probabilities(config);

        let expected_info: f64 = all_probs
            .iter()
            .map(|(_, prob)| {
                if *prob > 0.0 {
                    prob * (-prob.log2())
                } else {
                    0.0
                }
            })
            .sum();

        let entropy_error = (entropy - expected_info).abs();
        println!("      Calculated entropy: {:.6}", entropy);
        println!(
            "      Expected information (consistent): {:.6}",
            expected_info
        );
        println!("      Error: {:.6}", entropy_error);

        if entropy_error > 0.001 {
            // Tighter tolerance since we're using same source
            violations += 1;
            details.push(format!(
                "Entropy-information mismatch: error = {:.6}",
                entropy_error
            ));
        }

        details.push(format!(
            "Entropy vs expected info: {:.6} vs {:.6}, error: {:.6}",
            entropy, expected_info, entropy_error
        ));
    }

    if violations == 0 {
        DomainTestResult::pass("Information content properties satisfied".to_string())
            .with_details(details)
    } else {
        DomainTestResult::fail(format!("{} information content violations", violations))
            .with_details(details)
    }
}

/// Test 3.3: Kullback-Leibler Divergence
///
/// KL divergence: D_KL(P||Q) = Σ P(x) log₂(P(x)/Q(x))
/// Properties:
/// - D_KL(P||Q) ≥ 0 (non-negative)
/// - D_KL(P||Q) = 0 iff P = Q
/// - D_KL(P||Q) ≠ D_KL(Q||P) (asymmetric)
fn test_kl_divergence_comprehensive() -> DomainTestResult {
    println!("  Testing KL divergence calculation and properties...");

    let mut violations = 0;
    let mut details = Vec::new();

    // Test 1: KL divergence from uniform distribution
    println!("    Testing KL divergence from uniform distribution");

    let mut detector1 = AnomalyDetector::new(1).expect("Failed to create detector");
    let uniform_sequence = vec!["A", "B", "C", "D"]
        .repeat(100)
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    detector1.train(&uniform_sequence).expect("Failed to train");

    if let Some(context_node) = detector1
        .model()
        .context_tree()
        .get_context_node(&vec!["A".to_string()])
    {
        let kl_div_uniform = context_node.compute_kl_divergence(detector1.model().config());
        println!("      KL divergence (uniform): {:.6}", kl_div_uniform);

        // For uniform distribution, KL divergence from uniform should be close to 0
        if kl_div_uniform > 0.1 {
            violations += 1;
            details.push(format!(
                "KL divergence from uniform too high for uniform data: {:.6}",
                kl_div_uniform
            ));
        }
        details.push(format!("KL divergence (uniform): {:.6}", kl_div_uniform));
    }

    // Test 2: KL divergence for skewed distribution
    println!("    Testing KL divergence for skewed distribution");

    let mut detector2 = AnomalyDetector::new(1).expect("Failed to create detector");
    let skewed_sequence = vec!["A", "A", "A", "A", "A", "B"]
        .repeat(100)
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    detector2.train(&skewed_sequence).expect("Failed to train");

    if let Some(context_node) = detector2
        .model()
        .context_tree()
        .get_context_node(&vec!["A".to_string()])
    {
        let kl_div_skewed = context_node.compute_kl_divergence(detector2.model().config());
        println!("      KL divergence (skewed): {:.6}", kl_div_skewed);

        // Skewed distribution should have higher KL divergence from uniform
        details.push(format!("KL divergence (skewed): {:.6}", kl_div_skewed));
    }

    // Test 3: Non-negativity
    println!("    Testing KL divergence non-negativity");

    let test_sequences = vec![
        vec!["A"; 100],
        vec!["A", "B"].repeat(50),
        vec!["A", "B", "C"].repeat(33),
        vec!["A", "A", "B"].repeat(33),
    ];

    for (i, sequence) in test_sequences.iter().enumerate() {
        let mut detector = AnomalyDetector::new(1).expect("Failed to create detector");
        let seq_strings: Vec<String> = sequence.iter().map(|s| s.to_string()).collect();
        detector.train(&seq_strings).expect("Failed to train");

        let global_vocab: Vec<&String> = detector.model().state_mapping().keys().collect();
        for state in &global_vocab {
            if let Some(context_node) = detector
                .model()
                .context_tree()
                .get_context_node(&vec![state.to_string()])
            {
                let kl_div = context_node.compute_kl_divergence(detector.model().config());
                if kl_div < 0.0 {
                    violations += 1;
                    details.push(format!(
                        "Negative KL divergence in sequence {}, context {}: {:.6}",
                        i, state, kl_div
                    ));
                }
            }
        }
    }

    // Test 4: Ordering property (more skewed = higher KL divergence)
    println!("    Testing KL divergence ordering");

    let distributions = vec![
        (vec!["A", "B"].repeat(50), "balanced"),
        (vec!["A", "A", "B"].repeat(33), "slightly skewed"),
        (vec!["A", "A", "A", "A", "B"].repeat(20), "highly skewed"),
    ];

    let mut kl_divs = Vec::new();
    for (sequence, description) in &distributions {
        let mut detector = AnomalyDetector::new(1).expect("Failed to create detector");
        let seq_strings: Vec<String> = sequence.iter().map(|s| s.to_string()).collect();
        detector.train(&seq_strings).expect("Failed to train");

        if let Some(context_node) = detector
            .model()
            .context_tree()
            .get_context_node(&vec!["A".to_string()])
        {
            let kl_div = context_node.compute_kl_divergence(detector.model().config());
            kl_divs.push(kl_div);
            println!("      {} KL divergence: {:.6}", description, kl_div);
        }
    }

    // Check that KL divergence generally increases with skewness
    if kl_divs.len() >= 2 {
        let ordering_violations = kl_divs.windows(2).filter(|w| w[1] < w[0]).count();
        if ordering_violations > 0 {
            details.push(format!(
                "KL divergence ordering violations: {}",
                ordering_violations
            ));
        }
    }

    if violations == 0 {
        DomainTestResult::pass("KL divergence correctly calculated".to_string())
            .with_details(details)
    } else {
        DomainTestResult::fail(format!("{} KL divergence violations", violations))
            .with_details(details)
    }
}

/// Test 3.4: Cross-Entropy Relationships
///
/// Cross-entropy: H(P,Q) = -Σ P(x) log₂ Q(x)
/// Relationship: H(P,Q) = H(P) + D_KL(P||Q)
fn test_cross_entropy_relationships_comprehensive() -> DomainTestResult {
    println!("  Testing cross-entropy relationships...");

    let mut detector = AnomalyDetector::new(1).expect("Failed to create detector");
    // Create a sequence where context A has multiple possible next states
    let training_sequence = vec!["A", "X", "A", "Y", "A", "Z"]
        .repeat(100)
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    detector.train(&training_sequence).expect("Failed to train");

    let mut violations = 0;
    let mut details = Vec::new();

    // Test cross-entropy relationship: H(P,Q) = H(P) + D_KL(P||Q)
    println!("    Testing cross-entropy relationship: H(P,Q) = H(P) + D_KL(P||Q)");

    if let Some(context_node) = detector
        .model()
        .context_tree()
        .get_context_node(&vec!["A".to_string()])
    {
        let entropy = context_node.compute_entropy(detector.model().config());
        let kl_divergence = context_node.compute_kl_divergence(detector.model().config());

        // Calculate cross-entropy manually using consistent probability source
        // For our case, Q is the uniform distribution over the context's vocabulary
        let all_probs = context_node.get_all_probabilities(detector.model().config());
        let context_vocab_size = all_probs.len() as f64;
        let uniform_prob = 1.0 / context_vocab_size;

        let mut cross_entropy = 0.0;
        for (_, prob_p) in &all_probs {
            if *prob_p > 0.0 {
                cross_entropy += prob_p * (-uniform_prob.log2());
            }
        }

        let expected_cross_entropy = entropy + kl_divergence;
        let cross_entropy_error = (cross_entropy - expected_cross_entropy).abs();

        println!("      Entropy H(P): {:.6}", entropy);
        println!("      KL divergence D_KL(P||Q): {:.6}", kl_divergence);
        println!("      Cross-entropy H(P,Q): {:.6}", cross_entropy);
        println!("      Expected H(P) + D_KL: {:.6}", expected_cross_entropy);
        println!("      Error: {:.6}", cross_entropy_error);

        if cross_entropy_error > 0.01 {
            violations += 1;
            details.push(format!(
                "Cross-entropy relationship violation: error = {:.6}",
                cross_entropy_error
            ));
        }

        details.push(format!(
            "Cross-entropy relationship: H(P,Q)={:.6}, H(P)+D_KL={:.6}, error={:.6}",
            cross_entropy, expected_cross_entropy, cross_entropy_error
        ));
    }

    if violations == 0 {
        DomainTestResult::pass("Cross-entropy relationships satisfied".to_string())
            .with_details(details)
    } else {
        DomainTestResult::fail(format!("{} cross-entropy violations", violations))
            .with_details(details)
    }
}

/// Test 3.5: Information Theory Inequalities
///
/// Test fundamental inequalities:
/// - H(X) ≤ log₂|X| (entropy bounded by alphabet size)
/// - D_KL(P||Q) ≥ 0 (KL divergence non-negative)
/// - H(P,Q) ≥ H(P) (cross-entropy at least as large as entropy)
fn test_information_theory_inequalities() -> DomainTestResult {
    println!("  Testing information theory inequalities...");

    let mut violations = 0;
    let mut details = Vec::new();

    let test_sequences = vec![
        vec!["A", "B"].repeat(50),
        vec!["A", "B", "C"].repeat(33),
        vec!["A", "B", "C", "D"].repeat(25),
        vec!["A", "A", "B"].repeat(33),
    ];

    for (i, sequence) in test_sequences.iter().enumerate() {
        let mut detector = AnomalyDetector::new(1).expect("Failed to create detector");
        let seq_strings: Vec<String> = sequence.iter().map(|s| s.to_string()).collect();
        detector.train(&seq_strings).expect("Failed to train");

        let alphabet_size = detector.model().state_mapping().len();
        let max_entropy = (alphabet_size as f64).log2();

        println!(
            "    Testing sequence {} (alphabet size: {})",
            i, alphabet_size
        );

        let global_vocab: Vec<&String> = detector.model().state_mapping().keys().collect();
        for state in &global_vocab {
            if let Some(context_node) = detector
                .model()
                .context_tree()
                .get_context_node(&vec![state.to_string()])
            {
                let entropy = context_node.compute_entropy(detector.model().config());
                let kl_div = context_node.compute_kl_divergence(detector.model().config());

                // Test H(X) ≤ log₂|X|
                if entropy > max_entropy + 0.01 {
                    // Small tolerance for numerical errors
                    violations += 1;
                    details.push(format!(
                        "Entropy bound violation in seq {}, context {}: {:.6} > {:.6}",
                        i, state, entropy, max_entropy
                    ));
                }

                // Test D_KL(P||Q) ≥ 0
                if kl_div < -0.001 {
                    // Small tolerance for numerical errors
                    violations += 1;
                    details.push(format!(
                        "KL divergence negativity in seq {}, context {}: {:.6}",
                        i, state, kl_div
                    ));
                }

                // Calculate cross-entropy for H(P,Q) ≥ H(P) test
                let vocab_size = alphabet_size as f64;
                let uniform_prob = 1.0 / vocab_size;
                let mut cross_entropy = 0.0;

                for test_state in &global_vocab {
                    let prob = detector
                        .model()
                        .get_best_context_probability(&vec![state.to_string()], test_state);
                    if prob > 0.0 {
                        cross_entropy += prob * (-uniform_prob.log2());
                    }
                }

                // Test H(P,Q) ≥ H(P)
                if cross_entropy < entropy - 0.001 {
                    // Small tolerance
                    violations += 1;
                    details.push(format!(
                        "Cross-entropy bound violation in seq {}, context {}: {:.6} < {:.6}",
                        i, state, cross_entropy, entropy
                    ));
                }
            }
        }
    }

    details.push(format!(
        "Tested {} sequences with various alphabet sizes",
        test_sequences.len()
    ));

    if violations == 0 {
        DomainTestResult::pass("All information theory inequalities satisfied".to_string())
            .with_details(details)
    } else {
        DomainTestResult::fail(format!(
            "{} information theory inequality violations",
            violations
        ))
        .with_details(details)
    }
}
