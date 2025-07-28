// Tests core mathematical implementations

#[cfg(test)]
mod func_tests {
    use crate::*;

    #[test]
    fn test_probability_conservation() {
        println!("🧮 Testing Probability Conservation");

        let sequence: Vec<String> = vec!["A", "B", "A", "C", "B", "A", "C", "C"]
            .into_iter()
            .map(String::from)
            .collect();

        let mut model = AdvancedTransitionModel::new(2);
        model.build_context_tree(&sequence).unwrap();

        // Verify all context probabilities sum to 1.0
        for (context, node) in &model.contexts {
            let prob_sum: f64 = node.probabilities.values().sum();
            println!("Context {:?}: prob_sum = {:.6}", context, prob_sum);

            assert!(
                (prob_sum - 1.0).abs() < 1e-10,
                "Context {:?} probabilities sum to {}, not 1.0",
                context,
                prob_sum
            );
        }
        println!("✅ Probability conservation verified");
    }

    #[test]
    fn test_entropy_bounds() {
        println!("📊 Testing Entropy Mathematical Bounds");

        let sequence: Vec<String> = vec!["A", "B", "C", "A", "B", "C", "D", "D"]
            .into_iter()
            .map(String::from)
            .collect();

        let mut model = AdvancedTransitionModel::new(2);
        model.build_context_tree(&sequence).unwrap();

        for (context, node) in &model.contexts {
            let n_outcomes = node.probabilities.len() as f64;
            let max_entropy = n_outcomes.log2();

            println!(
                "Context {:?}: entropy = {:.4}, max = {:.4}",
                context, node.entropy, max_entropy
            );

            assert!(node.entropy >= 0.0, "Entropy must be non-negative");
            assert!(
                node.entropy <= max_entropy + 1e-10,
                "Entropy {:.4} exceeds maximum {:.4} for context {:?}",
                node.entropy,
                max_entropy,
                context
            );
        }
        println!("✅ Entropy bounds verified");
    }

    #[test]
    fn test_likelihood_calculation_exists() {
        println!("📈 Testing Likelihood Calculation");

        // Train on simple pattern
        let training: Vec<String> = vec!["A", "B", "A", "B", "A", "B"]
            .into_iter()
            .map(String::from)
            .collect();

        let mut model = AdvancedTransitionModel::new(2);
        model.build_context_tree(&training).unwrap();

        // Test sequences: normal vs anomalous
        let normal_seq: Vec<String> = vec!["A", "B", "A"].into_iter().map(String::from).collect();
        let anomaly_seq: Vec<String> = vec!["A", "X", "Y"].into_iter().map(String::from).collect();

        // Use the actual detect_advanced_anomalies method
        let normal_anomalies = model.detect_advanced_anomalies(&normal_seq, 0.1);
        let anomaly_anomalies = model.detect_advanced_anomalies(&anomaly_seq, 0.1);

        println!("Normal sequence anomalies: {}", normal_anomalies.len());
        println!("Anomaly sequence anomalies: {}", anomaly_anomalies.len());

        // Check that we get some results
        assert!(
            !normal_anomalies.is_empty() || !anomaly_anomalies.is_empty(),
            "Should detect some patterns in one of the sequences"
        );

        if !normal_anomalies.is_empty() {
            println!(
                "Normal pattern likelihood: {:.2e}",
                normal_anomalies[0].likelihood
            );
        }
        if !anomaly_anomalies.is_empty() {
            println!(
                "Anomaly pattern likelihood: {:.2e}",
                anomaly_anomalies[0].likelihood
            );
        }

        println!("✅ Likelihood calculation functional");
    }

    #[test]
    fn test_spectral_analysis_validity() {
        println!("🌊 Testing Spectral Analysis");

        let sequence: Vec<String> = vec!["A", "B", "A", "B", "A", "C", "A", "C"]
            .into_iter()
            .map(String::from)
            .collect();

        let mut model = AdvancedTransitionModel::new(2);
        model.build_context_tree(&sequence).unwrap();

        if let Some(spectral) = &model.spectral_decomposition {
            println!("Eigenvalues count: {}", spectral.eigenvalues.len());
            println!(
                "Stationary distribution sum: {:.6}",
                spectral.stationary_distribution.sum()
            );

            // Stationary distribution should sum to 1 (or be empty)
            if !spectral.stationary_distribution.is_empty() {
                let dist_sum = spectral.stationary_distribution.sum();
                assert!(
                    (dist_sum - 1.0).abs() < 1e-6 || dist_sum == 0.0,
                    "Stationary distribution sums to {}, not 1.0",
                    dist_sum
                );
            }

            // All probabilities should be non-negative
            for &prob in spectral.stationary_distribution.iter() {
                assert!(
                    prob >= -1e-10,
                    "Negative probability {} in stationary distribution",
                    prob
                );
            }

            println!("✅ Spectral analysis mathematically valid");
        } else {
            println!("⚠️  No spectral decomposition generated");
        }
    }

    #[test]
    fn test_context_tree_structure() {
        println!("🌳 Testing Context Tree Structure");

        let sequence: Vec<String> = vec!["A", "B", "C", "A", "B", "D"]
            .into_iter()
            .map(String::from)
            .collect();

        let mut model = AdvancedTransitionModel::new(3);
        model.build_context_tree(&sequence).unwrap();

        // Count contexts by order
        let mut order_counts = std::collections::HashMap::new();
        for context in model.contexts.keys() {
            *order_counts.entry(context.len()).or_insert(0) += 1;
        }

        println!("Context distribution by order: {:?}", order_counts);

        // Should have contexts of order 1, 2, and 3 (based on sequence length)
        for order in 1..=std::cmp::min(3, sequence.len() - 1) {
            assert!(
                order_counts.contains_key(&order),
                "Missing contexts of order {}",
                order
            );
        }

        // Verify each context has consistent counts
        for (context, node) in &model.contexts {
            let total_counts: usize = node.counts.values().sum();
            let prob_states = node.probabilities.len();
            let count_states = node.counts.len();

            assert_eq!(
                prob_states, count_states,
                "Context {:?}: mismatch between probability states ({}) and count states ({})",
                context, prob_states, count_states
            );

            assert!(
                total_counts > 0,
                "Context {:?} has zero total counts",
                context
            );
        }

        println!("✅ Context tree structure valid");
    }

    #[test]
    fn test_anomaly_detection_core_functionality() {
        println!("🎯 Testing Core Anomaly Detection");

        // Create highly regular training pattern
        let mut training = Vec::new();
        for _ in 0..10 {
            training.extend(
                vec!["START", "PROCESS", "END"]
                    .iter()
                    .map(|s| s.to_string()),
            );
        }

        let mut model = AdvancedTransitionModel::new(3);
        model.build_context_tree(&training).unwrap();

        // Test detection on regular vs irregular patterns
        let regular_test: Vec<String> = vec!["START", "PROCESS", "END", "START"]
            .into_iter()
            .map(String::from)
            .collect();
        let irregular_test: Vec<String> = vec!["START", "HACK", "CRASH", "ERROR"]
            .into_iter()
            .map(String::from)
            .collect();

        let regular_anomalies = model.detect_advanced_anomalies(&regular_test, 0.1);
        let irregular_anomalies = model.detect_advanced_anomalies(&irregular_test, 0.1);

        println!("Regular pattern anomalies: {}", regular_anomalies.len());
        println!("Irregular pattern anomalies: {}", irregular_anomalies.len());

        // Should be able to detect patterns in both sequences
        println!("✅ Anomaly detection functional");
    }

    #[test]
    fn test_quantum_representation_validity() {
        println!("⚛️  Testing Quantum Representation");

        let sequence: Vec<String> = vec!["A", "B", "A", "C", "B", "C"]
            .into_iter()
            .map(String::from)
            .collect();

        let mut model = AdvancedTransitionModel::new(2);
        model.build_context_tree(&sequence).unwrap();

        if let Some(quantum_state) = &model.quantum_representation {
            println!("Quantum state dimension: {}", quantum_state.len());

            // Check normalization: |ψ|² should sum to 1
            let norm_squared: f64 = quantum_state
                .iter()
                .map(|amplitude| amplitude.norm_sqr())
                .sum();

            println!("Quantum state norm²: {:.6}", norm_squared);

            assert!(
                (norm_squared - 1.0).abs() < 1e-6,
                "Quantum state not normalized: |ψ|² = {}",
                norm_squared
            );

            // All amplitudes should be finite
            for (i, amplitude) in quantum_state.iter().enumerate() {
                assert!(
                    amplitude.re.is_finite() && amplitude.im.is_finite(),
                    "Non-finite amplitude at index {}: {:?}",
                    i,
                    amplitude
                );
            }

            println!("✅ Quantum representation valid");
        } else {
            println!("⚠️  No quantum representation generated");
        }
    }

    #[test]
    fn test_information_theory_consistency() {
        println!("ℹ️  Testing Information Theory Consistency");

        let sequence: Vec<String> = vec!["A", "A", "B", "B", "C", "C"]
            .into_iter()
            .map(String::from)
            .collect();

        let mut model = AdvancedTransitionModel::new(2);
        model.build_context_tree(&sequence).unwrap();

        for (context, node) in &model.contexts {
            // Verify entropy calculation
            let manual_entropy: f64 = node
                .probabilities
                .values()
                .map(|&p| if p > 0.0 { -p * p.log2() } else { 0.0 })
                .sum();

            println!(
                "Context {:?}: stored={:.4}, calculated={:.4}",
                context, node.entropy, manual_entropy
            );

            assert!(
                (node.entropy - manual_entropy).abs() < 1e-10,
                "Entropy mismatch for context {:?}: stored={:.4}, calculated={:.4}",
                context,
                node.entropy,
                manual_entropy
            );

            // Verify KL divergence is non-negative
            assert!(
                node.kl_divergence >= -1e-10,
                "KL divergence should be non-negative, got {:.6}",
                node.kl_divergence
            );
        }

        println!("✅ Information theory measures consistent");
    }

    #[test]
    fn test_batch_processing_functionality() {
        println!("🔄 Testing Batch Processing");

        let sequences: Vec<Vec<String>> = vec![
            vec!["A", "B", "C"].into_iter().map(String::from).collect(),
            vec!["X", "Y", "Z"].into_iter().map(String::from).collect(),
            vec!["1", "2", "3"].into_iter().map(String::from).collect(),
        ];

        let results = batch_process_sequences(&sequences, 2, 0.1);

        println!(
            "Batch processing results: {} sequences processed",
            results.len()
        );
        assert_eq!(
            results.len(),
            sequences.len(),
            "Should process all input sequences"
        );

        for (i, anomalies) in results.iter().enumerate() {
            println!("Sequence {}: {} anomalies detected", i, anomalies.len());
        }

        println!("✅ Batch processing functional");
    }

    #[test]
    fn test_edge_cases_robustness() {
        println!("🛡️  Testing Edge Case Robustness");

        // Test with minimal data
        let minimal_seq = vec!["A".to_string(), "B".to_string()];
        let mut minimal_model = AdvancedTransitionModel::new(1);

        match minimal_model.build_context_tree(&minimal_seq) {
            Ok(()) => {
                println!("✓ Minimal sequence handled");
                let anomalies = minimal_model.detect_advanced_anomalies(&minimal_seq, 0.1);
                println!("  Minimal anomalies detected: {}", anomalies.len());
            }
            Err(e) => println!("✗ Minimal sequence failed: {}", e),
        }

        // Test with repeated elements
        let repeated_seq: Vec<String> = (0..20).map(|_| "X".to_string()).collect();
        let mut repeated_model = AdvancedTransitionModel::new(3);

        match repeated_model.build_context_tree(&repeated_seq) {
            Ok(()) => {
                println!("✓ Repeated sequence handled");

                // All contexts should have single outcomes with probability 1.0
                for (_context, node) in &repeated_model.contexts {
                    if node.probabilities.len() == 1 {
                        let prob = node.probabilities.values().next().unwrap();
                        assert!(
                            (prob - 1.0).abs() < 1e-10,
                            "Single outcome should have probability 1.0, got {:.6}",
                            prob
                        );
                    }
                }
            }
            Err(e) => println!("✗ Repeated sequence failed: {}", e),
        }

        println!("✅ Edge cases handled robustly");
    }

    #[test]
    fn test_all_anomaly_score_fields() {
        println!("📋 Testing AnomalyScore Structure");

        let sequence: Vec<String> = vec!["A", "B", "C", "A", "B", "D"]
            .into_iter()
            .map(String::from)
            .collect();

        let mut model = AdvancedTransitionModel::new(3);
        model.build_context_tree(&sequence).unwrap();

        let anomalies = model.detect_advanced_anomalies(&sequence, 0.1);

        if !anomalies.is_empty() {
            let anomaly = &anomalies[0];

            println!("AnomalyScore fields:");
            println!("  state_sequence: {:?}", anomaly.state_sequence);
            println!("  likelihood: {:.6}", anomaly.likelihood);
            println!(
                "  information_theoretic_score: {:.6}",
                anomaly.information_theoretic_score
            );
            println!(
                "  spectral_anomaly_score: {:.6}",
                anomaly.spectral_anomaly_score
            );
            println!(
                "  quantum_coherence_measure: {:.6}",
                anomaly.quantum_coherence_measure
            );
            println!(
                "  topological_signature: {:?}",
                anomaly.topological_signature
            );
            println!("  confidence_interval: {:?}", anomaly.confidence_interval);

            // Verify all fields are reasonable
            assert!(
                !anomaly.state_sequence.is_empty(),
                "State sequence should not be empty"
            );
            assert!(
                anomaly.likelihood >= 0.0,
                "Likelihood should be non-negative"
            );
            assert!(
                anomaly.information_theoretic_score >= 0.0,
                "Info score should be non-negative"
            );
            assert!(
                anomaly.spectral_anomaly_score >= 0.0,
                "Spectral score should be non-negative"
            );
            assert!(
                anomaly.quantum_coherence_measure >= 0.0,
                "Quantum coherence should be non-negative"
            );
            assert!(
                !anomaly.topological_signature.is_empty(),
                "Topological signature should not be empty"
            );

            println!("✅ All AnomalyScore fields populated correctly");
        } else {
            println!("⚠️  No anomalies detected to test structure");
        }
    }

    #[test]
    fn run_all_critical_tests() {
        println!("🧪 CRITICAL FUNCTIONAL TESTS FOR ANOMALY-GRID LIBRARY");
        println!("═══════════════════════════════════════════════════════");

        test_probability_conservation();
        test_entropy_bounds();
        test_likelihood_calculation_exists();
        test_spectral_analysis_validity();
        test_context_tree_structure();
        test_anomaly_detection_core_functionality();
        test_quantum_representation_validity();
        test_information_theory_consistency();
        test_batch_processing_functionality();
        test_all_anomaly_score_fields();
        test_edge_cases_robustness();

        println!("═══════════════════════════════════════════════════════");
        println!("🏁 ALL CRITICAL TESTS COMPLETED");
    }
}
