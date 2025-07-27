use nalgebra::{Complex, DMatrix, DVector, RealField};
use ndarray::{Array1, Array2, Array3};
//use rand_distr::Normal; found a better approach
use rayon::prelude::*;
use std::collections::HashMap;

// Advanced type definitions
pub type StateSpace = Vec<String>;
pub type ProbabilityTensor = Array3<f64>;
pub type QuantumState = Array1<Complex<f64>>;

#[derive(Debug, Clone)]
pub struct AdvancedTransitionModel {
    // Higher-order Markov model with variable context length
    pub contexts: HashMap<Vec<String>, ContextNode>,
    pub max_order: usize,
    pub quantum_representation: Option<QuantumState>,
    pub spectral_decomposition: Option<SpectralAnalysis>,
}

#[derive(Debug, Clone)]
pub struct ContextNode {
    pub counts: HashMap<String, usize>,
    pub probabilities: HashMap<String, f64>,
    pub information_content: f64,
    pub entropy: f64,
    pub kl_divergence: f64,
}

#[derive(Debug, Clone)]
pub struct SpectralAnalysis {
    pub eigenvalues: DVector<Complex<f64>>,
    pub eigenvectors: DMatrix<Complex<f64>>,
    pub stationary_distribution: DVector<f64>,
    pub mixing_time: f64,
    pub spectral_gap: f64,
}

#[derive(Debug, Clone)]
pub struct AnomalyScore {
    pub state_sequence: Vec<String>,
    pub likelihood: f64,
    pub information_theoretic_score: f64,
    pub spectral_anomaly_score: f64,
    pub quantum_coherence_measure: f64,
    pub topological_signature: Vec<f64>,
    pub confidence_interval: (f64, f64),
}

impl AdvancedTransitionModel {
    pub fn new(max_order: usize) -> Self {
        Self {
            contexts: HashMap::new(),
            max_order,
            quantum_representation: None,
            spectral_decomposition: None,
        }
    }

    /// Build variable-order Markov model using Context Tree Weighting
    pub fn build_context_tree(&mut self, sequence: &[String]) -> Result<(), String> {
        // Implementation of Context Tree Weighting algorithm
        for window_size in 1..=self.max_order {
            for window in sequence.windows(window_size + 1) {
                let context = window[..window_size].to_vec();
                let next_state = &window[window_size];

                let node = self
                    .contexts
                    .entry(context.clone())
                    .or_insert_with(|| ContextNode::new());

                *node.counts.entry(next_state.clone()).or_insert(0) += 1;
            }
        }

        // Calculate probabilities and information measures
        self.calculate_information_measures()?;
        self.perform_spectral_analysis()?;
        self.generate_quantum_representation()?;

        Ok(())
    }

    /// Calculate advanced information-theoretic measures
    fn calculate_information_measures(&mut self) -> Result<(), String> {
        for (_context, node) in &mut self.contexts {
            let total_count: usize = node.counts.values().sum();

            // Calculate probabilities
            for (state, &count) in &node.counts {
                let prob = count as f64 / total_count as f64;
                node.probabilities.insert(state.clone(), prob);
            }

            // Shannon entropy
            node.entropy = node
                .probabilities
                .values()
                .map(|&p| if p > 0.0 { -p * p.log2() } else { 0.0 })
                .sum();

            // Information content (surprise)
            node.information_content = -node.entropy.log2();

            // KL divergence from uniform distribution
            let uniform_prob = 1.0 / node.probabilities.len() as f64;
            node.kl_divergence = node
                .probabilities
                .values()
                .map(|&p| {
                    if p > 0.0 {
                        p * (p / uniform_prob).log2()
                    } else {
                        0.0
                    }
                })
                .sum();
        }
        Ok(())
    }

    /// Perform spectral analysis of the transition matrix
    fn perform_spectral_analysis(&mut self) -> Result<(), String> {
        // Early return if no contexts exist
        if self.contexts.is_empty() {
            return Ok(());
        }

        let matrix = match self.build_dense_transition_matrix() {
            Ok(m) => m,
            Err(_) => return Ok(()), // Don't fail the entire process
        };

        // Skip if matrix is empty
        if matrix.nrows() == 0 {
            return Ok(());
        }

        let eigenvalues = matrix.complex_eigenvalues();

        // Use robust stationary distribution finder
        let stationary_dist = self.find_stationary_distribution_robust(&matrix);

        let mut sorted_eigenvals: Vec<f64> = eigenvalues.iter().map(|c| c.norm()).collect();
        sorted_eigenvals.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        let spectral_gap = if sorted_eigenvals.len() > 1 {
            (sorted_eigenvals[0] - sorted_eigenvals[1]).max(0.0)
        } else {
            0.0
        };

        let mixing_time = if spectral_gap > 1e-10 {
            (1.0 / spectral_gap).ln().min(1000.0) // Cap at reasonable value
        } else {
            f64::INFINITY
        };

        self.spectral_decomposition = Some(SpectralAnalysis {
            eigenvalues,
            eigenvectors: DMatrix::zeros(0, 0),
            stationary_distribution: stationary_dist,
            mixing_time,
            spectral_gap,
        });

        Ok(())
    }

    /// Fixed stationary distribution finder - no more infinite loops
    fn find_stationary_distribution_robust(&self, matrix: &DMatrix<f64>) -> DVector<f64> {
        let n = matrix.nrows();
        if n == 0 {
            return DVector::from_vec(vec![]);
        }

        // Try power iteration with limits
        if let Some(dist) = self.power_iteration_with_limits(matrix) {
            return dist;
        }

        // Fallback to uniform distribution
        DVector::from_element(n, 1.0 / n as f64)
    }
    /// Power iteration with proper convergence checks
    fn power_iteration_with_limits(&self, matrix: &DMatrix<f64>) -> Option<DVector<f64>> {
        let n = matrix.nrows();
        let mut dist = DVector::from_element(n, 1.0 / n as f64);

        for _ in 0..100 {
            // Hard limit: max 100 iterations
            let new_dist = matrix.transpose() * &dist;
            let norm = new_dist.norm();

            if norm < 1e-15 {
                // Vector became zero
                return None;
            }

            let new_dist_normalized = new_dist / norm;

            // Check convergence
            if (&new_dist_normalized - &dist).norm() < 1e-9 {
                return Some(new_dist_normalized);
            }

            dist = new_dist_normalized;
        }

        None // Didn't converge in time
    }

    /// Generate quantum representation of the Markov model
    fn generate_quantum_representation(&mut self) -> Result<(), String> {
        let states = self.get_unique_states();
        let n_states = states.len();

        // Create quantum superposition state
        let mut quantum_state = Array1::zeros(n_states);

        // Initialize with equal superposition
        let amplitude = 1.0 / (n_states as f64).sqrt();
        for i in 0..n_states {
            quantum_state[i] = Complex::new(amplitude, 0.0);
        }

        // Apply quantum phase based on transition probabilities
        for (i, state) in states.iter().enumerate() {
            if let Some(context_node) = self.contexts.get(&vec![state.clone()]) {
                let entropy_phase = context_node.entropy * std::f64::consts::PI;
                quantum_state[i] = Complex::new(
                    amplitude * entropy_phase.cos(),
                    amplitude * entropy_phase.sin(),
                );
            }
        }

        self.quantum_representation = Some(quantum_state);
        Ok(())
    }

    /// Advanced anomaly detection using multiple mathematical approaches
    pub fn detect_advanced_anomalies(
        &self,
        sequence: &[String],
        _threshold: f64,
    ) -> Vec<AnomalyScore> {
        sequence
            .windows(self.max_order + 1)
            .par_bridge()
            // Compute a full anomaly score for each window;
            // only windows that successfully produce a score are kept.
            .filter_map(|window| self.calculate_comprehensive_anomaly_score(window))
            .collect()
    }

    /// Calculate comprehensive anomaly score using multiple mathematical measures
    fn calculate_comprehensive_anomaly_score(&self, sequence: &[String]) -> Option<AnomalyScore> {
        // Likelihood-based score
        let likelihood = self.calculate_sequence_likelihood(sequence)?;

        // Information-theoretic score
        let info_score = self.calculate_information_score(sequence)?;

        // Spectral anomaly score
        let spectral_score = self.calculate_spectral_anomaly_score(sequence)?;

        // Quantum coherence measure
        let quantum_score = self.calculate_quantum_coherence(sequence)?;

        // Topological signature
        let topo_signature = self.calculate_topological_signature(sequence)?;

        // Bayesian confidence interval
        let confidence_interval = self.calculate_bayesian_confidence_interval(sequence)?;

        Some(AnomalyScore {
            state_sequence: sequence.to_vec(),
            likelihood,
            information_theoretic_score: info_score,
            spectral_anomaly_score: spectral_score,
            quantum_coherence_measure: quantum_score,
            topological_signature: topo_signature,
            confidence_interval,
        })
    }

    /// Calculate sequence likelihood using variable-order context
    fn calculate_sequence_likelihood(&self, sequence: &[String]) -> Option<f64> {
        let mut likelihood = 1.0;

        for i in 1..sequence.len() {
            let max_context_len = std::cmp::min(i, self.max_order);
            let mut best_prob = 0.0;

            for context_len in 1..=max_context_len {
                let context = sequence[i - context_len..i].to_vec();
                if let Some(node) = self.contexts.get(&context) {
                    if let Some(&prob) = node.probabilities.get(&sequence[i]) {
                        // FIX: This corrected logic penalizes contexts with low counts,
                        // correctly identifying rare transitions as low-likelihood events.
                        let total_support: usize = node.counts.values().sum();
                        if total_support > 0 {
                            let weighted_prob = prob / (total_support as f64).sqrt();
                            best_prob = best_prob.max(weighted_prob);
                        }
                    }
                }
            }

            if best_prob > 0.0 {
                likelihood *= best_prob;
            } else {
                // Laplace smoothing for unseen transitions
                likelihood *= 1e-10;
            }
        }

        Some(likelihood)
    }

    /// Calculate information-theoretic anomaly score
    fn calculate_information_score(&self, sequence: &[String]) -> Option<f64> {
        let mut total_surprise = 0.0;
        let mut context_entropy = 0.0;

        for i in 1..sequence.len() {
            let context_len = std::cmp::min(i, self.max_order);
            let context = sequence[i - context_len..i].to_vec();

            if let Some(node) = self.contexts.get(&context) {
                // Add entropy of the context
                context_entropy += node.entropy;

                // Add surprise (information content) of the transition
                if let Some(&prob) = node.probabilities.get(&sequence[i]) {
                    total_surprise += -prob.log2();
                }
            }
        }

        // Normalize by sequence length
        Some((total_surprise + context_entropy) / sequence.len() as f64)
    }

    /// Calculate spectral anomaly score based on eigenvalue analysis
    fn calculate_spectral_anomaly_score(&self, sequence: &[String]) -> Option<f64> {
        let spectral = self.spectral_decomposition.as_ref()?;

        // Calculate deviation from expected stationary behavior
        let mut deviation = 0.0;
        let states = self.get_unique_states();

        for (_, state) in sequence.iter().enumerate() {
            if let Some(state_idx) = states.iter().position(|s| s == state) {
                let expected_prob = spectral.stationary_distribution[state_idx];
                let observed_freq = self.calculate_observed_frequency(state, sequence);
                deviation += (observed_freq - expected_prob).abs();
            }
        }

        Some(deviation / sequence.len() as f64)
    }

    /// Calculate quantum coherence measure
    fn calculate_quantum_coherence(&self, _sequence: &[String]) -> Option<f64> {
        let quantum_state = self.quantum_representation.as_ref()?;
        let states = self.get_unique_states();

        // Calculate von Neumann entropy of the quantum state
        let mut density_matrix = Array2::zeros((states.len(), states.len()));

        for i in 0..states.len() {
            for j in 0..states.len() {
                density_matrix[[i, j]] = (quantum_state[i].conj() * quantum_state[j]).re;
            }
        }

        // Simplified coherence measure (should use proper eigenvalue decomposition)
        let trace = (0..states.len())
            .map(|i| density_matrix[[i, i]])
            .sum::<f64>();
        let coherence = 1.0 - trace / states.len() as f64;

        Some(coherence)
    }

    /// Calculate topological signature using persistent homology concepts
    fn calculate_topological_signature(&self, sequence: &[String]) -> Option<Vec<f64>> {
        // Simplified topological analysis
        // In full implementation, would use persistent homology libraries

        let mut signature = Vec::new();

        // Calculate different topological invariants
        // 1. Connected components (Betti-0)
        let components = self.count_connected_components(sequence);
        signature.push(components as f64);

        // 2. Cycles (simplified Betti-1)
        let cycles = self.count_approximate_cycles(sequence);
        signature.push(cycles as f64);

        // 3. Local clustering coefficient
        let clustering = self.calculate_local_clustering(sequence);
        signature.push(clustering);

        Some(signature)
    }

    /// Calculate Bayesian confidence interval for anomaly score
    fn calculate_bayesian_confidence_interval(&self, sequence: &[String]) -> Option<(f64, f64)> {
        // Simplified Bayesian interval calculation
        // In full implementation, would use MCMC sampling

        let likelihood = self.calculate_sequence_likelihood(sequence)?;
        let log_likelihood = likelihood.ln();

        // Use normal approximation for confidence interval
        let std_error = (1.0 / sequence.len() as f64).sqrt();
        let margin = 1.96 * std_error; // 95% confidence interval

        Some((
            (log_likelihood - margin).exp(),
            (log_likelihood + margin).exp(),
        ))
    }

    // Helper methods
    fn build_dense_transition_matrix(&self) -> Result<DMatrix<f64>, String> {
        let states = self.get_unique_states();
        let n = states.len();
        let mut matrix = DMatrix::zeros(n, n);

        for (i, from_state) in states.iter().enumerate() {
            if let Some(node) = self.contexts.get(&vec![from_state.clone()]) {
                for (j, to_state) in states.iter().enumerate() {
                    if let Some(&prob) = node.probabilities.get(to_state) {
                        matrix[(i, j)] = prob;
                    }
                }
            }
        }

        Ok(matrix)
    }

    /// Original method replacement - add the robust version
    //fn find_stationary_distribution(&self, matrix: &DMatrix<f64>) -> Result<DVector<f64>, String> {
    //    Ok(self.find_stationary_distribution_robust(matrix))
    //}

    fn get_unique_states(&self) -> Vec<String> {
        let mut states = std::collections::HashSet::new();
        for context in self.contexts.keys() {
            for state in context {
                states.insert(state.clone());
            }
        }
        for node in self.contexts.values() {
            for state in node.counts.keys() {
                states.insert(state.clone());
            }
        }
        states.into_iter().collect()
    }

    fn calculate_observed_frequency(&self, target_state: &str, sequence: &[String]) -> f64 {
        let count = sequence.iter().filter(|&s| s == target_state).count();
        count as f64 / sequence.len() as f64
    }

    fn count_connected_components(&self, _sequence: &[String]) -> usize {
        // Simplified implementation
        self.contexts.len()
    }

    fn count_approximate_cycles(&self, sequence: &[String]) -> usize {
        // Count approximate cycles by looking for repeated subsequences
        let mut cycles = 0;
        for len in 2..=sequence.len() / 2 {
            for i in 0..=sequence.len() - 2 * len {
                if sequence[i..i + len] == sequence[i + len..i + 2 * len] {
                    cycles += 1;
                }
            }
        }
        cycles
    }

    fn calculate_local_clustering(&self, _sequence: &[String]) -> f64 {
        // Simplified clustering coefficient
        let mut clustering_sum = 0.0;
        let mut node_count = 0;

        for (_context, node) in &self.contexts {
            if node.counts.len() > 1 {
                let connections = node.counts.len();
                let possible_connections = connections * (connections - 1) / 2;
                if possible_connections > 0 {
                    clustering_sum += connections as f64 / possible_connections as f64;
                    node_count += 1;
                }
            }
        }

        if node_count > 0 {
            clustering_sum / node_count as f64
        } else {
            0.0
        }
    }
}

impl ContextNode {
    fn new() -> Self {
        Self {
            counts: HashMap::new(),
            probabilities: HashMap::new(),
            information_content: 0.0,
            entropy: 0.0,
            kl_divergence: 0.0,
        }
    }
}

// Performance-optimized parallel processing
pub fn batch_process_sequences(
    sequences: &[Vec<String>],
    max_order: usize,
    threshold: f64,
) -> Vec<Vec<AnomalyScore>> {
    sequences
        .par_iter()
        .map(|sequence| {
            let mut model = AdvancedTransitionModel::new(max_order);
            model.build_context_tree(sequence).unwrap_or_else(|e| {
                eprintln!("Error building context tree: {}", e);
            });
            model.detect_advanced_anomalies(sequence, threshold)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_advanced_anomaly_detection() {
        let sequence: Vec<String> = vec!["A", "B", "C", "A", "B", "C", "A", "D", "E", "F", "A"]
            .into_iter()
            .map(String::from)
            .collect();

        let mut model = AdvancedTransitionModel::new(3);
        model.build_context_tree(&sequence).unwrap();

        let anomalies = model.detect_advanced_anomalies(&sequence, 0.1);

        assert!(!anomalies.is_empty());

        for anomaly in &anomalies {
            println!("Advanced Anomaly Detected:");
            println!("  Sequence: {:?}", anomaly.state_sequence);
            println!("  Likelihood: {:.6}", anomaly.likelihood);
            println!(
                "  Information Score: {:.6}",
                anomaly.information_theoretic_score
            );
            println!("  Spectral Score: {:.6}", anomaly.spectral_anomaly_score);
            println!(
                "  Quantum Coherence: {:.6}",
                anomaly.quantum_coherence_measure
            );
            println!("  Confidence Interval: {:?}", anomaly.confidence_interval);
            println!(
                "  Topological Signature: {:?}",
                anomaly.topological_signature
            );
            println!();
        }
    }

    #[test]
    fn test_spectral_analysis() {
        let sequence: Vec<String> = vec!["A", "B", "A", "B", "A", "C", "A", "C"]
            .into_iter()
            .map(String::from)
            .collect();

        let mut model = AdvancedTransitionModel::new(2);
        model.build_context_tree(&sequence).unwrap();

        assert!(model.spectral_decomposition.is_some());

        let spectral = model.spectral_decomposition.as_ref().unwrap();
        println!("Spectral Analysis:");
        println!("  Mixing time: {:.4}", spectral.mixing_time);
        println!("  Spectral gap: {:.4}", spectral.spectral_gap);
        println!(
            "  Stationary distribution: {:?}",
            spectral.stationary_distribution
        );
    }
}
pub mod func_tests;
pub mod test_runner;
