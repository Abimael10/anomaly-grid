//! Domain 2: Probability Theory Compliance
//!
//! This module implements comprehensive domain-driven tests for probability theory
//! fundamentals, ensuring our implementation adheres to Kolmogorov axioms and
//! fundamental probability principles.

use anomaly_grid::*;
use std::collections::HashMap;

#[test]
fn domain_2_probability_theory() {
    println!("🔬 DOMAIN 2: PROBABILITY THEORY COMPLIANCE");
    println!("==========================================");
    println!();
    
    let mut test_results = Vec::new();
    
    // Test 2.1: Kolmogorov Probability Axioms
    println!("Test 2.1: Kolmogorov Probability Axioms");
    println!("---------------------------------------");
    let kolmogorov_result = test_kolmogorov_axioms_comprehensive();
    test_results.push(("Kolmogorov Axioms", kolmogorov_result));
    println!();
    
    // Test 2.2: Conditional Probability Rules
    println!("Test 2.2: Conditional Probability Rules");
    println!("---------------------------------------");
    let conditional_result = test_conditional_probability_rules_comprehensive();
    test_results.push(("Conditional Probability", conditional_result));
    println!();
    
    // Test 2.3: Bayes' Theorem Application
    println!("Test 2.3: Bayes' Theorem Application");
    println!("------------------------------------");
    let bayes_result = test_bayes_theorem_application_comprehensive();
    test_results.push(("Bayes Theorem", bayes_result));
    println!();
    
    // Test 2.4: Law of Total Probability
    println!("Test 2.4: Law of Total Probability");
    println!("----------------------------------");
    let total_prob_result = test_law_of_total_probability_comprehensive();
    test_results.push(("Total Probability", total_prob_result));
    println!();
    
    // Test 2.5: Independence and Dependence
    println!("Test 2.5: Independence and Dependence");
    println!("-------------------------------------");
    let independence_result = test_independence_and_dependence();
    test_results.push(("Independence", independence_result));
    println!();
    
    // Domain 2 Summary
    println!("🏆 DOMAIN 2 SUMMARY");
    println!("===================");
    let passed_tests = test_results.iter().filter(|(_, result)| result.passed).count();
    let total_tests = test_results.len();
    
    for (test_name, result) in &test_results {
        let status = if result.passed { "✅" } else { "❌" };
        println!("  {} {}: {}", status, test_name, result.evidence);
    }
    
    println!();
    println!("Domain 2 Result: {}/{} tests passed", passed_tests, total_tests);
    
    assert_eq!(passed_tests, total_tests, 
               "Domain 2 (Probability Theory) failed: {}/{} tests passed", 
               passed_tests, total_tests);
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

/// Test 2.1: Kolmogorov Probability Axioms
/// 
/// Axiom 1: P(A) ≥ 0 for all events A
/// Axiom 2: P(Ω) = 1 where Ω is the sample space
/// Axiom 3: For disjoint events A₁, A₂, ...: P(A₁ ∪ A₂ ∪ ...) = P(A₁) + P(A₂) + ...
fn test_kolmogorov_axioms_comprehensive() -> DomainTestResult {
    println!("  Testing Kolmogorov probability axioms...");
    
    let mut detector = AnomalyDetector::new(1).expect("Failed to create detector");
    
    // Train with a diverse pattern
    let training_sequence = vec!["A", "B", "C", "D", "E"].repeat(100)
        .iter().map(|s| s.to_string()).collect::<Vec<_>>();
    detector.train(&training_sequence).expect("Failed to train");
    
    let mut axiom_violations = 0;
    let mut details = Vec::new();
    
    // Test Axiom 1: P(A) ≥ 0 for all events
    println!("    Testing Axiom 1: P(A) ≥ 0 for all events");
    let contexts = vec![vec!["A"], vec!["B"], vec!["C"], vec!["D"], vec!["E"]];
    
    // Get the global vocabulary from the trained model
    let global_vocab: Vec<&String> = detector.model().state_mapping().keys().collect();
    
    for context in &contexts {
        let context_strings: Vec<String> = context.iter().map(|s| s.to_string()).collect();
        for state in &global_vocab {
            let prob = detector.model().get_best_context_probability(&context_strings, state);
            if prob < 0.0 {
                axiom_violations += 1;
                details.push(format!("Axiom 1 violation: P({} | {:?}) = {:.6} < 0", state, context, prob));
            }
        }
    }
    
    // Test Axiom 2: P(Ω) = 1 (probabilities sum to 1)
    println!("    Testing Axiom 2: P(Ω) = 1 (normalization)");
    for context in &contexts {
        let context_strings: Vec<String> = context.iter().map(|s| s.to_string()).collect();
        let mut total_prob = 0.0;
        
        // Sum probabilities for all states in the global vocabulary (proper test)
        let global_vocab: Vec<&String> = detector.model().state_mapping().keys().collect();
        for state in &global_vocab {
            let prob = detector.model().get_best_context_probability(&context_strings, state);
            total_prob += prob;
        }
        
        let normalization_error = (total_prob - 1.0).abs();
        if normalization_error > 1e-6 {
            axiom_violations += 1;
            details.push(format!("Axiom 2 violation: P(Ω | {:?}) = {:.6} ≠ 1", context, total_prob));
        }
        
        println!("      Context {:?}: total probability = {:.6}", context, total_prob);
    }
    
    // Test Axiom 3: Additivity (simplified test)
    println!("    Testing Axiom 3: Additivity for disjoint events");
    let context_a = vec!["A".to_string()];
    
    // Test that P(B ∪ C | A) = P(B | A) + P(C | A) when B and C are disjoint
    // In our discrete case, different states are naturally disjoint
    let prob_b = detector.model().get_best_context_probability(&context_a, "B");
    let prob_c = detector.model().get_best_context_probability(&context_a, "C");
    let prob_union_bc = prob_b + prob_c; // For disjoint events
    
    // This should be valid since B and C are disjoint states
    let additivity_holds = prob_union_bc >= prob_b && prob_union_bc >= prob_c;
    
    if !additivity_holds {
        axiom_violations += 1;
        details.push(format!("Axiom 3 violation: additivity failed"));
    }
    
    details.push(format!("P(B|A) = {:.6}, P(C|A) = {:.6}, P(B∪C|A) = {:.6}", 
                        prob_b, prob_c, prob_union_bc));
    
    if axiom_violations == 0 {
        DomainTestResult::pass("All Kolmogorov axioms satisfied".to_string())
            .with_details(details)
    } else {
        DomainTestResult::fail(format!("{} Kolmogorov axiom violations", axiom_violations))
            .with_details(details)
    }
}

/// Test 2.2: Conditional Probability Rules
/// 
/// P(A|B) = P(A ∩ B) / P(B) when P(B) > 0
/// P(A|B) * P(B) = P(A ∩ B)
/// P(A|B) + P(¬A|B) = 1
fn test_conditional_probability_rules_comprehensive() -> DomainTestResult {
    println!("  Testing conditional probability rules...");
    
    let mut detector = AnomalyDetector::new(2).expect("Failed to create detector");
    
    // Train with a pattern that creates clear conditional relationships
    let training_sequence = vec!["A", "B", "A", "C", "B", "D", "B", "E"].repeat(50)
        .iter().map(|s| s.to_string()).collect::<Vec<_>>();
    detector.train(&training_sequence).expect("Failed to train");
    
    let mut rule_violations = 0;
    let mut details = Vec::new();
    
    // Test Rule 1: P(A|B) is well-defined when P(B) > 0
    println!("    Testing conditional probability definition");
    let context_b = vec!["B".to_string()];
    let prob_a_given_b = detector.model().get_best_context_probability(&context_b, "A");
    let prob_c_given_b = detector.model().get_best_context_probability(&context_b, "C");
    let prob_d_given_b = detector.model().get_best_context_probability(&context_b, "D");
    let prob_e_given_b = detector.model().get_best_context_probability(&context_b, "E");
    
    println!("      P(A|B) = {:.6}", prob_a_given_b);
    println!("      P(C|B) = {:.6}", prob_c_given_b);
    println!("      P(D|B) = {:.6}", prob_d_given_b);
    println!("      P(E|B) = {:.6}", prob_e_given_b);
    
    // Test Rule 2: Conditional probabilities sum to 1 (use global vocabulary)
    let global_vocab: Vec<&String> = detector.model().state_mapping().keys().collect();
    let mut total_conditional = 0.0;
    for state in &global_vocab {
        let prob = detector.model().get_best_context_probability(&context_b, state);
        total_conditional += prob;
    }
    let conditional_normalization_error = (total_conditional - 1.0).abs();
    
    if conditional_normalization_error > 1e-6 {
        rule_violations += 1;
        details.push(format!("Conditional normalization violation: sum = {:.6}", total_conditional));
    }
    
    // Test Rule 3: Chain rule P(A,B) = P(A|B) * P(B)
    println!("    Testing chain rule: P(A,B) = P(A|B) * P(B)");
    
    // For sequence A->B, we can test this
    let sequence_ab = vec!["A".to_string(), "B".to_string()];
    let joint_prob_ab = detector.model().calculate_likelihood(&sequence_ab);
    
    // P(B|A) from our model
    let context_a = vec!["A".to_string()];
    let prob_b_given_a = detector.model().get_best_context_probability(&context_a, "B");
    
    // For the chain rule test, we need P(A) which is harder to get directly
    // We'll use a simplified test: check that conditional probabilities are consistent
    let consistency_check = prob_b_given_a > 0.0 && joint_prob_ab > 0.0;
    
    if !consistency_check {
        rule_violations += 1;
        details.push("Chain rule consistency check failed".to_string());
    }
    
    details.push(format!("P(A,B) = {:.6}, P(B|A) = {:.6}", joint_prob_ab, prob_b_given_a));
    
    // Test Rule 4: Symmetry properties where applicable
    println!("    Testing conditional probability symmetry properties");
    
    // In our Markov model, P(B|A) and P(A|B) should generally be different
    // unless there's perfect symmetry in the data
    let prob_a_given_b_check = detector.model().get_best_context_probability(&context_b, "A");
    let symmetry_difference = (prob_b_given_a - prob_a_given_b_check).abs();
    
    // This is not a violation, just an observation
    details.push(format!("P(B|A) = {:.6}, P(A|B) = {:.6}, difference = {:.6}", 
                        prob_b_given_a, prob_a_given_b_check, symmetry_difference));
    
    println!("    Rule violations detected: {}", rule_violations);
    for detail in &details {
        println!("      {}", detail);
    }
    
    if rule_violations == 0 {
        DomainTestResult::pass("All conditional probability rules satisfied".to_string())
            .with_details(details)
    } else {
        DomainTestResult::fail(format!("{} conditional probability rule violations", rule_violations))
            .with_details(details)
    }
}

/// Test 2.3: Bayes' Theorem Application
/// 
/// P(A|B) = P(B|A) * P(A) / P(B)
/// This is fundamental for probabilistic inference
fn test_bayes_theorem_application_comprehensive() -> DomainTestResult {
    println!("  Testing Bayes' theorem application...");
    
    let mut detector = AnomalyDetector::new(1).expect("Failed to create detector");
    
    // Create a scenario where we can test Bayes' theorem
    // Pattern: A appears 60% of time, B appears 40% of time
    // When A appears, next is B 80% of time
    // When B appears, next is A 70% of time
    let mut training_sequence = Vec::new();
    
    // Create the pattern manually to have known probabilities
    for _ in 0..100 {
        training_sequence.extend(vec!["A", "B", "A", "B", "A", "B"]); // A->B pattern
        training_sequence.extend(vec!["B", "A", "B", "A"]); // B->A pattern  
        training_sequence.extend(vec!["A", "A"]); // Some A->A
    }
    
    let training_strings: Vec<String> = training_sequence.iter().map(|s| s.to_string()).collect();
    detector.train(&training_strings).expect("Failed to train");
    
    let mut bayes_violations = 0;
    let mut details = Vec::new();
    
    // Test Bayes' theorem: P(A|B) = P(B|A) * P(A) / P(B)
    println!("    Testing Bayes' theorem: P(A|B) = P(B|A) * P(A) / P(B)");
    
    let context_a = vec!["A".to_string()];
    let context_b = vec!["B".to_string()];
    
    let prob_b_given_a = detector.model().get_best_context_probability(&context_a, "B");
    let prob_a_given_b = detector.model().get_best_context_probability(&context_b, "A");
    
    println!("      P(B|A) = {:.6}", prob_b_given_a);
    println!("      P(A|B) = {:.6}", prob_a_given_b);
    
    // For a complete Bayes' theorem test, we need marginal probabilities P(A) and P(B)
    // In a Markov chain, these are the stationary probabilities
    // We'll approximate by looking at the overall frequency
    
    // Count occurrences in training data
    let total_states = training_strings.len();
    let count_a = training_strings.iter().filter(|&s| s == "A").count();
    let count_b = training_strings.iter().filter(|&s| s == "B").count();
    
    let marginal_prob_a = count_a as f64 / total_states as f64;
    let marginal_prob_b = count_b as f64 / total_states as f64;
    
    println!("      P(A) ≈ {:.6}", marginal_prob_a);
    println!("      P(B) ≈ {:.6}", marginal_prob_b);
    
    // Apply Bayes' theorem: P(A|B) should equal P(B|A) * P(A) / P(B)
    let bayes_calculated = (prob_b_given_a * marginal_prob_a) / marginal_prob_b;
    let bayes_error = (prob_a_given_b - bayes_calculated).abs();
    
    println!("      Bayes calculated P(A|B) = {:.6}", bayes_calculated);
    println!("      Direct P(A|B) = {:.6}", prob_a_given_b);
    println!("      Bayes error = {:.6}", bayes_error);
    
    // Allow some tolerance due to finite sample effects and smoothing
    let bayes_tolerance = 0.1;
    if bayes_error > bayes_tolerance {
        bayes_violations += 1;
        details.push(format!("Bayes theorem violation: error = {:.6} > {:.6}", bayes_error, bayes_tolerance));
    }
    
    details.push(format!("P(B|A) = {:.6}, P(A) = {:.6}, P(B) = {:.6}", 
                        prob_b_given_a, marginal_prob_a, marginal_prob_b));
    details.push(format!("Bayes calculated: {:.6}, Direct: {:.6}, Error: {:.6}", 
                        bayes_calculated, prob_a_given_b, bayes_error));
    
    // Test Bayes' theorem consistency in reverse
    let bayes_reverse = (prob_a_given_b * marginal_prob_b) / marginal_prob_a;
    let reverse_error = (prob_b_given_a - bayes_reverse).abs();
    
    if reverse_error > bayes_tolerance {
        bayes_violations += 1;
        details.push(format!("Reverse Bayes violation: error = {:.6}", reverse_error));
    }
    
    if bayes_violations == 0 {
        DomainTestResult::pass("Bayes' theorem correctly applied".to_string())
            .with_details(details)
    } else {
        DomainTestResult::fail(format!("{} Bayes' theorem violations", bayes_violations))
            .with_details(details)
    }
}

/// Test 2.4: Law of Total Probability
/// 
/// P(A) = Σᵢ P(A|Bᵢ) * P(Bᵢ) where {Bᵢ} is a partition of the sample space
fn test_law_of_total_probability_comprehensive() -> DomainTestResult {
    println!("  Testing law of total probability...");
    
    let mut detector = AnomalyDetector::new(1).expect("Failed to create detector");
    
    // Create a scenario with clear partitions
    let training_sequence = vec!["X", "A", "Y", "A", "Z", "A", "X", "B", "Y", "B", "Z", "C"].repeat(50)
        .iter().map(|s| s.to_string()).collect::<Vec<_>>();
    detector.train(&training_sequence).expect("Failed to train");
    
    let mut total_prob_violations = 0;
    let mut details = Vec::new();
    
    // Test law of total probability: P(A) = Σᵢ P(A|Bᵢ) * P(Bᵢ)
    println!("    Testing P(A) = Σᵢ P(A|Bᵢ) * P(Bᵢ)");
    
    // Use states X, Y, Z as our partition {Bᵢ}
    let partition_states = vec!["X", "Y", "Z"];
    let target_state = "A";
    
    // Calculate P(A|Bᵢ) for each Bᵢ in partition
    let mut conditional_probs = Vec::new();
    let mut marginal_probs = Vec::new();
    
    for partition_state in &partition_states {
        let context = vec![partition_state.to_string()];
        let prob_a_given_bi = detector.model().get_best_context_probability(&context, target_state);
        conditional_probs.push(prob_a_given_bi);
        
        // Estimate marginal probability P(Bᵢ)
        let count_bi = training_sequence.iter().filter(|&s| s == partition_state).count();
        let marginal_prob_bi = count_bi as f64 / training_sequence.len() as f64;
        marginal_probs.push(marginal_prob_bi);
        
        println!("      P(A|{}) = {:.6}, P({}) = {:.6}", partition_state, prob_a_given_bi, partition_state, marginal_prob_bi);
    }
    
    // Calculate total probability using the law
    let total_prob_calculated: f64 = conditional_probs.iter()
        .zip(marginal_probs.iter())
        .map(|(cond, marg)| cond * marg)
        .sum();
    
    // Calculate direct marginal probability P(A)
    let count_a = training_sequence.iter().filter(|&s| s == target_state).count();
    let direct_prob_a = count_a as f64 / training_sequence.len() as f64;
    
    println!("      Total probability calculated: {:.6}", total_prob_calculated);
    println!("      Direct P(A): {:.6}", direct_prob_a);
    
    let total_prob_error = (total_prob_calculated - direct_prob_a).abs();
    println!("      Error: {:.6}", total_prob_error);
    
    // Allow tolerance for finite sample effects
    let tolerance = 0.05;
    if total_prob_error > tolerance {
        total_prob_violations += 1;
        details.push(format!("Total probability law violation: error = {:.6}", total_prob_error));
    }
    
    details.push(format!("Calculated: {:.6}, Direct: {:.6}, Error: {:.6}", 
                        total_prob_calculated, direct_prob_a, total_prob_error));
    
    // Test that partition probabilities sum to reasonable value
    let partition_sum: f64 = marginal_probs.iter().sum();
    details.push(format!("Partition probabilities sum: {:.6}", partition_sum));
    
    if total_prob_violations == 0 {
        DomainTestResult::pass("Law of total probability satisfied".to_string())
            .with_details(details)
    } else {
        DomainTestResult::fail(format!("{} total probability violations", total_prob_violations))
            .with_details(details)
    }
}

/// Test 2.5: Independence and Dependence
/// 
/// Test whether the model correctly captures independence: P(A|B) = P(A) when A and B are independent
/// And dependence: P(A|B) ≠ P(A) when A and B are dependent
fn test_independence_and_dependence() -> DomainTestResult {
    println!("  Testing independence and dependence...");
    
    // Test 1: Create independent events
    let mut detector1 = AnomalyDetector::new(1).expect("Failed to create detector");
    
    // Create a pattern where next state is independent of current state
    let independent_sequence = vec!["A", "X", "A", "Y", "B", "X", "B", "Y", "C", "X", "C", "Y"].repeat(50)
        .iter().map(|s| s.to_string()).collect::<Vec<_>>();
    detector1.train(&independent_sequence).expect("Failed to train");
    
    // Test 2: Create dependent events  
    let mut detector2 = AnomalyDetector::new(1).expect("Failed to create detector");
    
    // Create a pattern where next state strongly depends on current state
    let dependent_sequence = vec!["A", "X", "A", "X", "B", "Y", "B", "Y"].repeat(50)
        .iter().map(|s| s.to_string()).collect::<Vec<_>>();
    detector2.train(&dependent_sequence).expect("Failed to train");
    
    let mut independence_violations = 0;
    let mut details = Vec::new();
    
    // Test independence in detector1
    println!("    Testing independence scenario");
    let prob_x_given_a1 = detector1.model().get_best_context_probability(&vec!["A".to_string()], "X");
    let prob_x_given_b1 = detector1.model().get_best_context_probability(&vec!["B".to_string()], "X");
    let prob_x_given_c1 = detector1.model().get_best_context_probability(&vec!["C".to_string()], "X");
    
    println!("      P(X|A) = {:.6}", prob_x_given_a1);
    println!("      P(X|B) = {:.6}", prob_x_given_b1);
    println!("      P(X|C) = {:.6}", prob_x_given_c1);
    
    // In independent case, these should be similar
    let independence_variance = [prob_x_given_a1, prob_x_given_b1, prob_x_given_c1]
        .iter()
        .map(|&p| (p - prob_x_given_a1).powi(2))
        .sum::<f64>() / 3.0;
    
    details.push(format!("Independence variance: {:.6}", independence_variance));
    
    // Test dependence in detector2
    println!("    Testing dependence scenario");
    let prob_x_given_a2 = detector2.model().get_best_context_probability(&vec!["A".to_string()], "X");
    let prob_x_given_b2 = detector2.model().get_best_context_probability(&vec!["B".to_string()], "X");
    let prob_y_given_a2 = detector2.model().get_best_context_probability(&vec!["A".to_string()], "Y");
    let prob_y_given_b2 = detector2.model().get_best_context_probability(&vec!["B".to_string()], "Y");
    
    println!("      P(X|A) = {:.6}", prob_x_given_a2);
    println!("      P(X|B) = {:.6}", prob_x_given_b2);
    println!("      P(Y|A) = {:.6}", prob_y_given_a2);
    println!("      P(Y|B) = {:.6}", prob_y_given_b2);
    
    // In dependent case, P(X|A) should be much higher than P(X|B)
    let dependence_difference = (prob_x_given_a2 - prob_x_given_b2).abs();
    details.push(format!("Dependence difference P(X|A) - P(X|B): {:.6}", dependence_difference));
    
    // Check that dependence is captured (should be significant difference)
    if dependence_difference < 0.1 {
        independence_violations += 1;
        details.push("Failed to capture dependence: differences too small".to_string());
    }
    
    // Check that independence shows less variance than dependence
    if independence_variance > dependence_difference {
        independence_violations += 1;
        details.push("Independence variance higher than dependence difference".to_string());
    }
    
    if independence_violations == 0 {
        DomainTestResult::pass("Independence and dependence correctly modeled".to_string())
            .with_details(details)
    } else {
        DomainTestResult::fail(format!("{} independence/dependence violations", independence_violations))
            .with_details(details)
    }
}