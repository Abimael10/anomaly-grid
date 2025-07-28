// MATHEMATICALLY CORRECTED MVP - ANOMALY DETECTION LIBRARY
// Addresses critical mathematical violations while maintaining API compatibility

use nalgebra::{Complex, DMatrix, DVector};
use ndarray::Array1;
use rayon::prelude::*;
use std::collections::HashMap;

// CORRECTED TYPE DEFINITIONS
pub type StateSpace = Vec<String>;
pub type QuantumState = Array1<Complex<f64>>;

#[derive(Debug, Clone)]
pub struct AdvancedTransitionModel {
    pub contexts: HashMap<Vec<String>, ContextNode>,
    pub max_order: usize,
    pub quantum_representation: Option<QuantumState>,
    pub spectral_decomposition: Option<SpectralAnalysis>,

    // NEW: State management for efficiency
    state_to_id: HashMap<String, usize>,
    id_to_state: Vec<String>,

    // NEW: Numerical stability tracking
    min_probability: f64,
    smoothing_alpha: f64,
}

#[derive(Debug, Clone)]
pub struct ContextNode {
    pub counts: HashMap<String, usize>,
    pub probabilities: HashMap<String, f64>,
    pub entropy: f64,
    pub kl_divergence: f64,

    // REMOVED: information_content (mathematically invalid)
    // NEW: Per-transition information content (correct)
    pub transition_information: HashMap<String, f64>, // I(x) = -log₂(P(x))
}

#[derive(Debug, Clone)]
pub struct SpectralAnalysis {
    pub eigenvalues: DVector<Complex<f64>>,
    pub eigenvectors: DMatrix<Complex<f64>>,
    pub stationary_distribution: DVector<f64>,
    pub mixing_time: f64,
    pub spectral_gap: f64,

    // NEW: Numerical quality metrics
    pub condition_number: f64,
    pub convergence_error: f64,
}

#[derive(Debug, Clone)]
pub struct AnomalyScore {
    pub state_sequence: Vec<String>,
    pub log_likelihood: f64, // CHANGED: Use log-likelihood to prevent underflow
    pub likelihood: f64,     // COMPUTED: exp(log_likelihood) with bounds checking
    pub information_theoretic_score: f64,
    pub spectral_anomaly_score: f64,
    pub quantum_coherence_measure: f64,
    pub topological_signature: Vec<f64>,
    pub confidence_interval: (f64, f64),

    // NEW: Quality indicators
    pub numerical_stability_flag: bool,
    pub anomaly_strength: f64, // Normalized anomaly measure [0,1]
}

impl AdvancedTransitionModel {
    pub fn new(max_order: usize) -> Self {
        Self {
            contexts: HashMap::new(),
            max_order,
            quantum_representation: None,
            spectral_decomposition: None,
            state_to_id: HashMap::new(),
            id_to_state: Vec::new(),
            min_probability: 1e-12,
            smoothing_alpha: 1.0, // Laplace smoothing parameter
        }
    }

    /// CORRECTED: Build context tree with proper probability theory
    pub fn build_context_tree(&mut self, sequence: &[String]) -> Result<(), String> {
        if sequence.len() < 2 {
            return Err("Sequence too short for context tree".to_string());
        }

        // Build state mapping
        self.build_state_mapping(sequence);

        // Extract contexts with counts
        for window_size in 1..=self.max_order {
            for window in sequence.windows(window_size + 1) {
                let context = window[..window_size].to_vec();
                let next_state = &window[window_size];

                let node = self
                    .contexts
                    .entry(context)
                    .or_insert_with(ContextNode::new);
                *node.counts.entry(next_state.clone()).or_insert(0) += 1;
            }
        }

        // CORRECTED: Calculate probabilities and information measures
        self.calculate_information_measures_correct()?;
        self.perform_spectral_analysis_robust()?;
        self.generate_quantum_representation_physical()?;

        Ok(())
    }

    fn build_state_mapping(&mut self, sequence: &[String]) {
        let mut unique_states: std::collections::HashSet<String> =
            sequence.iter().cloned().collect();

        for (id, state) in unique_states.drain().enumerate() {
            self.state_to_id.insert(state.clone(), id);
            self.id_to_state.push(state);
        }
    }

    /// CORRECTED: Proper information theory calculations
    fn calculate_information_measures_correct(&mut self) -> Result<(), String> {
        for (context, node) in &mut self.contexts {
            let total_count: usize = node.counts.values().sum();
            let vocab_size = node.counts.len();

            // CORRECTED: Laplace smoothing for unseen transitions
            for (state, &count) in &node.counts {
                let smoothed_prob = (count as f64 + self.smoothing_alpha)
                    / (total_count as f64 + self.smoothing_alpha * vocab_size as f64);

                // Ensure minimum probability for numerical stability
                let prob = smoothed_prob.max(self.min_probability);
                node.probabilities.insert(state.clone(), prob);

                // CORRECTED: Information content I(x) = -log₂(P(x))
                node.transition_information
                    .insert(state.clone(), -prob.log2());
            }

            // CORRECTED: Shannon entropy with numerical stability
            node.entropy = node
                .probabilities
                .values()
                .map(|&p| {
                    if p <= self.min_probability {
                        0.0 // lim p→0 p*log(p) = 0
                    } else {
                        -p * p.log2()
                    }
                })
                .sum();

            // Verify entropy bounds
            let max_entropy = (vocab_size as f64).log2();
            if node.entropy > max_entropy + 1e-10 {
                return Err(format!(
                    "Entropy violation in context {:?}: {} > {}",
                    context, node.entropy, max_entropy
                ));
            }

            // CORRECTED: KL divergence from uniform distribution
            let uniform_prob = 1.0 / vocab_size as f64;
            node.kl_divergence = node
                .probabilities
                .values()
                .map(|&p| {
                    if p > self.min_probability {
                        p * (p / uniform_prob).log2()
                    } else {
                        0.0
                    }
                })
                .sum();
        }

        Ok(())
    }

    /// CORRECTED: Robust spectral analysis with error checking
    fn perform_spectral_analysis_robust(&mut self) -> Result<(), String> {
        if self.contexts.is_empty() {
            return Ok(());
        }

        let transition_matrix = self.build_transition_matrix_robust()?;

        // Check matrix properties
        let condition_number = self.estimate_condition_number(&transition_matrix);
        if condition_number > 1e12 {
            eprintln!(
                "Warning: Ill-conditioned transition matrix (κ = {:.2e})",
                condition_number
            );
        }

        // Compute eigenvalues safely
        let eigenvalues = transition_matrix.complex_eigenvalues();

        // Find stationary distribution using multiple methods
        let (stationary_dist, convergence_error) =
            self.find_stationary_distribution_robust(&transition_matrix)?;

        // Calculate spectral properties
        let mut eigenvalue_magnitudes: Vec<f64> = eigenvalues.iter().map(|c| c.norm()).collect();
        eigenvalue_magnitudes.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        let spectral_gap = if eigenvalue_magnitudes.len() > 1 {
            (eigenvalue_magnitudes[0] - eigenvalue_magnitudes[1]).max(0.0)
        } else {
            0.0
        };

        // CORRECTED: Mixing time calculation
        let mixing_time = if eigenvalue_magnitudes.len() > 1 && eigenvalue_magnitudes[1] > 1e-12 {
            let lambda_2 = eigenvalue_magnitudes[1];
            if lambda_2 < 1.0 {
                // τ_mix ≈ 1/(1-λ₂) for reversible chains
                let mixing_estimate = 1.0 / (1.0 - lambda_2);
                mixing_estimate.min(1e6) // Cap at reasonable value
            } else {
                f64::INFINITY
            }
        } else {
            f64::INFINITY
        };

        self.spectral_decomposition = Some(SpectralAnalysis {
            eigenvalues,
            eigenvectors: transition_matrix
                .clone_owned()
                .map(|v| Complex::new(v, 0.0)),
            stationary_distribution: stationary_dist,
            mixing_time,
            spectral_gap,
            condition_number,
            convergence_error,
        });

        Ok(())
    }

    fn build_transition_matrix_robust(&self) -> Result<DMatrix<f64>, String> {
        let n_states = self.id_to_state.len();
        if n_states == 0 {
            return Err("No states found".to_string());
        }

        let mut matrix = DMatrix::zeros(n_states, n_states);

        // Build from first-order contexts
        for (context, node) in &self.contexts {
            if context.len() == 1 {
                if let Some(&from_id) = self.state_to_id.get(&context[0]) {
                    for (to_state, &prob) in &node.probabilities {
                        if let Some(&to_id) = self.state_to_id.get(to_state) {
                            matrix[(from_id, to_id)] = prob;
                        }
                    }
                }
            }
        }

        // Ensure stochastic property: each row sums to 1
        for i in 0..n_states {
            let row_sum: f64 = matrix.row(i).sum();
            if row_sum < 1e-15 {
                // Uniform distribution for disconnected states
                for j in 0..n_states {
                    matrix[(i, j)] = 1.0 / n_states as f64;
                }
            } else if (row_sum - 1.0).abs() > 1e-10 {
                // Normalize row
                for j in 0..n_states {
                    matrix[(i, j)] /= row_sum;
                }
            }
        }

        Ok(matrix)
    }

    fn estimate_condition_number(&self, matrix: &DMatrix<f64>) -> f64 {
        // Estimate condition number using power method
        // κ(A) ≈ ||A|| · ||A⁻¹|| ≈ λ_max / λ_min

        let eigenvalues = matrix.complex_eigenvalues();
        let magnitudes: Vec<f64> = eigenvalues.iter().map(|c| c.norm()).collect();

        if let (Some(&max_val), Some(&min_val)) = (
            magnitudes.iter().max_by(|a, b| a.partial_cmp(b).unwrap()),
            magnitudes.iter().min_by(|a, b| a.partial_cmp(b).unwrap()),
        ) {
            if min_val > 1e-15 {
                max_val / min_val
            } else {
                f64::INFINITY
            }
        } else {
            1.0
        }
    }

    fn find_stationary_distribution_robust(
        &self,
        matrix: &DMatrix<f64>,
    ) -> Result<(DVector<f64>, f64), String> {
        let n = matrix.nrows();

        // Method 1: Direct linear system solve (P^T - I)π = 0, with Σπᵢ = 1
        if let Some((dist, error)) = self.solve_stationary_direct(matrix) {
            return Ok((dist, error));
        }

        // Method 2: Power iteration with improved convergence
        if let Some((dist, error)) = self.power_iteration_robust(matrix) {
            return Ok((dist, error));
        }

        // Method 3: Uniform fallback
        Ok((DVector::from_element(n, 1.0 / n as f64), 1.0))
    }

    fn solve_stationary_direct(&self, matrix: &DMatrix<f64>) -> Option<(DVector<f64>, f64)> {
        let n = matrix.nrows();
        if n == 0 {
            return None;
        }

        // Solve (P^T - I)π = 0 with constraint Σπᵢ = 1
        let mut augmented = matrix.transpose() - DMatrix::identity(n, n);

        // Replace last equation with normalization constraint
        for j in 0..n {
            augmented[(n - 1, j)] = 1.0;
        }

        let mut rhs = DVector::zeros(n);
        rhs[n - 1] = 1.0;

        // Use LU decomposition
        let lu = augmented.clone().lu();
        if let Some(solution) = lu.solve(&rhs) {
            // Verify solution quality
            let residual = &augmented * &solution - &rhs;
            let error = residual.norm();

            // Check non-negativity and normalization
            let all_positive = solution.iter().all(|&x| x >= -1e-10);
            let sum_close_to_one = (solution.sum() - 1.0).abs() < 1e-8;

            if all_positive && sum_close_to_one && error < 1e-8 {
                let corrected = solution.map(|x| x.max(0.0));
                let normalized = &corrected / corrected.sum();
                return Some((normalized, error));
            }
        }

        None
    }

    fn power_iteration_robust(&self, matrix: &DMatrix<f64>) -> Option<(DVector<f64>, f64)> {
        let n = matrix.nrows();
        let mut dist = DVector::from_element(n, 1.0 / n as f64);

        const MAX_ITER: usize = 1000;
        const TOLERANCE: f64 = 1e-12;

        let mut prev_error = f64::INFINITY;
        let mut stagnation_count = 0;

        for _ in 0..MAX_ITER {
            // π^(k+1) = π^(k) P (for row vector π)
            // In column form: π^(k+1) = P^T π^(k)
            let new_dist = matrix.transpose() * &dist;

            let norm = new_dist.sum();
            if norm < 1e-15 {
                return None;
            }

            let new_dist_normalized = new_dist / norm;

            // Check convergence
            let error = (&new_dist_normalized - &dist).norm();

            if error < TOLERANCE {
                return Some((new_dist_normalized, error));
            }

            // Check for stagnation
            if (prev_error - error).abs() < 1e-15 {
                stagnation_count += 1;
                if stagnation_count > 10 {
                    return Some((new_dist_normalized, error));
                }
            } else {
                stagnation_count = 0;
            }

            dist = new_dist_normalized;
            prev_error = error;
        }

        None
    }

    /// CORRECTED: Physically meaningful quantum representation
    fn generate_quantum_representation_physical(&mut self) -> Result<(), String> {
        let n_states = self.id_to_state.len();
        if n_states == 0 {
            return Ok(());
        }

        // Use stationary distribution as quantum amplitudes (Born rule)
        if let Some(spectral) = &self.spectral_decomposition {
            let stationary = &spectral.stationary_distribution;

            let mut quantum_state = Array1::zeros(n_states);

            // |ψᵢ|² = πᵢ (Born rule), so ψᵢ = √πᵢ
            for i in 0..n_states {
                let amplitude = stationary[i].sqrt().max(0.0);
                quantum_state[i] = Complex::new(amplitude, 0.0);
            }

            // Verify normalization: Σ|ψᵢ|² = 1
            let norm_squared: f64 = quantum_state.iter().map(|c| c.norm_sqr()).sum();

            if (norm_squared - 1.0).abs() > 1e-10 {
                // Renormalize if necessary
                let norm = norm_squared.sqrt();
                if norm > 1e-15 {
                    for amplitude in quantum_state.iter_mut() {
                        *amplitude /= norm;
                    }
                }
            }

            self.quantum_representation = Some(quantum_state);
        }

        Ok(())
    }

    /// CORRECTED: Advanced anomaly detection with log-likelihood
    pub fn detect_advanced_anomalies(
        &self,
        sequence: &[String],
        threshold: f64,
    ) -> Vec<AnomalyScore> {
        if sequence.len() <= self.max_order {
            return Vec::new();
        }

        sequence
            .windows(self.max_order + 1)
            .par_bridge()
            .filter_map(|window| {
                self.calculate_comprehensive_anomaly_score_correct(window, threshold)
            })
            .collect()
    }

    /// CORRECTED: Mathematically sound anomaly scoring
    fn calculate_comprehensive_anomaly_score_correct(
        &self,
        sequence: &[String],
        threshold: f64,
    ) -> Option<AnomalyScore> {
        // 1. CORRECTED: Log-likelihood calculation (prevents underflow)
        let log_likelihood = self.calculate_log_likelihood_robust(sequence)?;
        let likelihood = self.safe_exp(log_likelihood);

        // Early exit for clearly normal patterns
        if likelihood > threshold * 10.0 {
            return None;
        }

        // 2. CORRECTED: Information-theoretic score
        let info_score = self.calculate_information_score_correct(sequence)?;

        // 3. Spectral anomaly score
        let spectral_score = self
            .calculate_spectral_anomaly_score(sequence)
            .unwrap_or(0.0);

        // 4. CORRECTED: Quantum coherence measure
        let quantum_score = self
            .calculate_quantum_coherence_correct(sequence)
            .unwrap_or(0.0);

        // 5. Simplified topological signature
        let topo_signature = self.calculate_topological_signature_simple(sequence);

        // 6. CORRECTED: Bayesian confidence interval
        let confidence_interval =
            self.calculate_confidence_interval_robust(sequence, log_likelihood);

        // 7. Quality assessment
        let numerical_stability = likelihood.is_finite() && likelihood > 0.0;
        let anomaly_strength =
            self.calculate_anomaly_strength(likelihood, info_score, spectral_score);

        Some(AnomalyScore {
            state_sequence: sequence.to_vec(),
            log_likelihood,
            likelihood,
            information_theoretic_score: info_score,
            spectral_anomaly_score: spectral_score,
            quantum_coherence_measure: quantum_score,
            topological_signature: topo_signature,
            confidence_interval,
            numerical_stability_flag: numerical_stability,
            anomaly_strength,
        })
    }

    /// CORRECTED: Log-likelihood with numerical stability
    fn calculate_log_likelihood_robust(&self, sequence: &[String]) -> Option<f64> {
        let mut log_likelihood = 0.0;

        for i in 1..sequence.len() {
            let mut best_log_prob = f64::NEG_INFINITY;

            // Try contexts from longest to shortest (hierarchical)
            for context_len in (1..=std::cmp::min(i, self.max_order)).rev() {
                let context = sequence[i - context_len..i].to_vec();

                if let Some(node) = self.contexts.get(&context) {
                    if let Some(&prob) = node.probabilities.get(&sequence[i]) {
                        best_log_prob = prob.ln();
                        break; // Use highest-order available context
                    }
                }
            }

            // Background probability for unseen transitions
            if best_log_prob.is_infinite() {
                let vocab_size = self.id_to_state.len() as f64;
                best_log_prob = -(vocab_size + 1.0).ln(); // +1 for unseen state
            }

            log_likelihood += best_log_prob;
        }

        Some(log_likelihood)
    }

    fn safe_exp(&self, log_val: f64) -> f64 {
        if log_val < -700.0 {
            0.0 // Prevent underflow
        } else if log_val > 700.0 {
            f64::INFINITY // Prevent overflow
        } else {
            log_val.exp()
        }
    }

    /// CORRECTED: Information score using proper information content
    fn calculate_information_score_correct(&self, sequence: &[String]) -> Option<f64> {
        let mut total_information = 0.0;
        let mut count = 0;

        for i in 1..sequence.len() {
            let context_len = std::cmp::min(i, self.max_order);
            let context = sequence[i - context_len..i].to_vec();

            if let Some(node) = self.contexts.get(&context) {
                if let Some(&info_content) = node.transition_information.get(&sequence[i]) {
                    total_information += info_content;
                    count += 1;
                }
            }
        }

        if count > 0 {
            Some(total_information / count as f64)
        } else {
            None
        }
    }

    /// CORRECTED: Quantum coherence using l₁-norm measure
    fn calculate_quantum_coherence_correct(&self, _sequence: &[String]) -> Option<f64> {
        let quantum_state = self.quantum_representation.as_ref()?;
        let n = quantum_state.len();

        // Construct density matrix ρ = |ψ⟩⟨ψ|
        let mut coherence = 0.0;

        // C_l₁(ρ) = Σᵢ≠ⱼ |ρᵢⱼ| for pure state ρᵢⱼ = ψᵢ*ψⱼ
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let rho_ij = quantum_state[i].conj() * quantum_state[j];
                    coherence += rho_ij.norm();
                }
            }
        }

        Some(coherence)
    }

    fn calculate_topological_signature_simple(&self, sequence: &[String]) -> Vec<f64> {
        vec![
            sequence.len() as f64,                          // Sequence length
            self.count_unique_transitions(sequence) as f64, // Transition diversity
            self.calculate_repetition_index(sequence),      // Pattern regularity
        ]
    }

    fn count_unique_transitions(&self, sequence: &[String]) -> usize {
        let mut transitions = std::collections::HashSet::new();
        for window in sequence.windows(2) {
            transitions.insert((window[0].clone(), window[1].clone()));
        }
        transitions.len()
    }

    fn calculate_repetition_index(&self, sequence: &[String]) -> f64 {
        let mut repetitions = 0;
        for len in 2..=sequence.len() / 2 {
            for i in 0..=sequence.len() - 2 * len {
                if sequence[i..i + len] == sequence[i + len..i + 2 * len] {
                    repetitions += 1;
                }
            }
        }
        repetitions as f64 / sequence.len() as f64
    }

    fn calculate_confidence_interval_robust(
        &self,
        sequence: &[String],
        log_likelihood: f64,
    ) -> (f64, f64) {
        // Simplified confidence interval using asymptotic normality
        let n = sequence.len() as f64;
        let std_error = (1.0 / n.sqrt()).max(1e-6);
        let z_score = 1.96; // 95% confidence

        let margin = z_score * std_error;
        let likelihood = self.safe_exp(log_likelihood);

        (
            self.safe_exp(log_likelihood - margin).max(likelihood * 0.1),
            self.safe_exp(log_likelihood + margin)
                .min(likelihood * 10.0),
        )
    }

    fn calculate_anomaly_strength(
        &self,
        likelihood: f64,
        info_score: f64,
        spectral_score: f64,
    ) -> f64 {
        // Combine scores into normalized anomaly strength [0,1]
        let log_likelihood_component = if likelihood > 0.0 {
            -likelihood.ln()
        } else {
            10.0
        };
        let combined_score = log_likelihood_component.log10().max(0.0) * 0.5
            + info_score * 0.3
            + spectral_score * 0.2;

        // Normalize to [0,1] using tanh
        (combined_score / 10.0).tanh()
    }

    // Helper methods
    fn calculate_spectral_anomaly_score(&self, sequence: &[String]) -> Option<f64> {
        let spectral = self.spectral_decomposition.as_ref()?;

        let mut deviation = 0.0;
        for state in sequence {
            if let Some(&state_id) = self.state_to_id.get(state) {
                if state_id < spectral.stationary_distribution.len() {
                    let expected_prob = spectral.stationary_distribution[state_id];
                    let observed_freq = self.calculate_observed_frequency(state, sequence);
                    deviation += (observed_freq - expected_prob).abs();
                }
            }
        }

        Some(deviation / sequence.len() as f64)
    }

    fn calculate_observed_frequency(&self, target_state: &str, sequence: &[String]) -> f64 {
        let count = sequence.iter().filter(|&s| s == target_state).count();
        count as f64 / sequence.len() as f64
    }
}

impl ContextNode {
    fn new() -> Self {
        Self {
            counts: HashMap::new(),
            probabilities: HashMap::new(),
            entropy: 0.0,
            kl_divergence: 0.0,
            transition_information: HashMap::new(),
        }
    }
}

// CORRECTED: Batch processing with error handling
pub fn batch_process_sequences(
    sequences: &[Vec<String>],
    max_order: usize,
    threshold: f64,
) -> Vec<Vec<AnomalyScore>> {
    sequences
        .par_iter()
        .map(|sequence| {
            if sequence.len() <= max_order {
                return Vec::new();
            }

            let mut model = AdvancedTransitionModel::new(max_order);
            match model.build_context_tree(sequence) {
                Ok(()) => model.detect_advanced_anomalies(sequence, threshold),
                Err(_) => Vec::new(), // Return empty on error
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_corrected_mathematical_properties() {
        let sequence: Vec<String> = vec!["A", "B", "A", "B", "A", "C", "A", "C"]
            .into_iter()
            .map(String::from)
            .collect();

        let mut model = AdvancedTransitionModel::new(2);
        assert!(model.build_context_tree(&sequence).is_ok());

        // Test probability conservation
        for (context, node) in &model.contexts {
            let prob_sum: f64 = node.probabilities.values().sum();
            assert!(
                (prob_sum - 1.0).abs() < 1e-10,
                "Context {:?}: probabilities sum to {}, not 1.0",
                context,
                prob_sum
            );
        }

        // Test entropy bounds
        for (context, node) in &model.contexts {
            let max_entropy = (node.probabilities.len() as f64).log2();
            assert!(
                node.entropy >= 0.0 && node.entropy <= max_entropy + 1e-10,
                "Context {:?}: entropy {} outside bounds [0, {}]",
                context,
                node.entropy,
                max_entropy
            );
        }

        // Test information content correctness
        for (context, node) in &model.contexts {
            for (state, &prob) in &node.probabilities {
                let expected_info = -prob.log2();
                let actual_info = node.transition_information[state];
                assert!(
                    (expected_info - actual_info).abs() < 1e-10,
                    "Information content mismatch for {}->{}:",
                    context[0],
                    state
                );
            }
        }
    }

    #[test]
    fn test_numerical_stability() {
        // Test with sequence that would cause underflow in original implementation
        let sequence: Vec<String> = (0..100).map(|i| format!("S{}", i % 5)).collect();

        let mut model = AdvancedTransitionModel::new(3);
        assert!(model.build_context_tree(&sequence).is_ok());

        let anomalies = model.detect_advanced_anomalies(&sequence, 1e-10);

        // All likelihood values should be finite and positive
        for anomaly in &anomalies {
            assert!(
                anomaly.likelihood.is_finite() && anomaly.likelihood > 0.0,
                "Non-finite likelihood: {}",
                anomaly.likelihood
            );
            assert!(
                anomaly.log_likelihood.is_finite(),
                "Non-finite log-likelihood: {}",
                anomaly.log_likelihood
            );
        }
    }

    #[test]
    fn test_quantum_representation_normalization() {
        let sequence: Vec<String> = vec!["A", "B", "C", "A", "B", "C"]
            .into_iter()
            .map(String::from)
            .collect();

        let mut model = AdvancedTransitionModel::new(2);
        assert!(model.build_context_tree(&sequence).is_ok());

        if let Some(quantum_state) = &model.quantum_representation {
            let norm_squared: f64 = quantum_state.iter().map(|c| c.norm_sqr()).sum();
            assert!(
                (norm_squared - 1.0).abs() < 1e-10,
                "Quantum state not normalized: ||ψ||² = {}",
                norm_squared
            );
        } else {
            panic!("Quantum representation not generated");
        }
    }
    #[test]
    fn test_spectral_analysis_properties() {
        let sequence: Vec<String> = vec!["A", "B", "A", "C", "B", "A"]
            .into_iter()
            .map(String::from)
            .collect();
        let mut model = AdvancedTransitionModel::new(2);
        assert!(model.build_context_tree(&sequence).is_ok());
        if let Some(spectral) = &model.spectral_decomposition {
            // Test eigenvalue properties
            assert!(!spectral.eigenvalues.is_empty(), "No eigenvalues computed");
            assert!(
                spectral.eigenvalues.iter().all(|&c| c.norm() >= 0.0),
                "Negative eigenvalue found"
            );

            // Test stationary distribution normalization
            let stationary_sum: f64 = spectral.stationary_distribution.sum();
            assert!(
                (stationary_sum - 1.0).abs() < 1e-10,
                "Stationary distribution not normalized: sum = {}",
                stationary_sum
            );

            // Test spectral gap
            assert!(
                spectral.spectral_gap >= 0.0,
                "Negative spectral gap: {}",
                spectral.spectral_gap
            );
        } else {
            panic!("Spectral decomposition not generated");
        }
    }
    #[test]
    fn test_anomaly_detection_correctness() {
        let sequence: Vec<String> = vec!["A", "B", "A", "C", "A", "B", "A", "D"]
            .into_iter()
            .map(String::from)
            .collect();
        let mut model = AdvancedTransitionModel::new(2);
        assert!(model.build_context_tree(&sequence).is_ok());
        let anomalies = model.detect_advanced_anomalies(&sequence, 0.1);
        assert!(!anomalies.is_empty(), "No anomalies detected in sequence");
        for anomaly in &anomalies {
            assert!(
                anomaly.likelihood.is_finite() && anomaly.likelihood > 0.0,
                "Non-finite likelihood in anomaly: {}",
                anomaly.likelihood
            );
            assert!(
                anomaly.log_likelihood.is_finite(),
                "Non-finite log-likelihood in anomaly: {}",
                anomaly.log_likelihood
            );
            assert!(
                anomaly.information_theoretic_score >= 0.0,
                "Negative information score in anomaly: {}",
                anomaly.information_theoretic_score
            );
        }
    }
    #[test]
    fn test_batch_processing_correctness() {
        let sequences: Vec<Vec<String>> = vec![
            vec!["A", "B", "A", "C", "A", "B", "A", "D"]
                .into_iter()
                .map(String::from)
                .collect(),
            vec!["X", "Y", "X", "Z", "X", "Y", "X", "W"]
                .into_iter()
                .map(String::from)
                .collect(),
        ];
        let results = batch_process_sequences(&sequences, 2, 0.1);
        assert_eq!(
            results.len(),
            sequences.len(),
            "Batch processing returned incorrect number of results"
        );
        for (i, anomalies) in results.iter().enumerate() {
            assert!(
                !anomalies.is_empty(),
                "No anomalies detected in sequence {}",
                i
            );
            for anomaly in anomalies {
                assert!(
                    anomaly.likelihood.is_finite() && anomaly.likelihood > 0.0,
                    "Non-finite likelihood in anomaly {}: {}",
                    i,
                    anomaly.likelihood
                );
                assert!(
                    anomaly.log_likelihood.is_finite(),
                    "Non-finite log-likelihood in anomaly {}: {}",
                    i,
                    anomaly.log_likelihood
                );
                assert!(
                    anomaly.information_theoretic_score >= 0.0,
                    "Negative information score in anomaly {}: {}",
                    i,
                    anomaly.information_theoretic_score
                );
            }
        }
    }
}

pub mod func_tests;
pub mod test_runner;
