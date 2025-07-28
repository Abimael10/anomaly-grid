# Advanced Markov Anomaly Detection Library
## Developer Manual v2.1

```
╔════════════════════════════════════════════════════╗
║                   anomaly-grid                     ║
║           Sequential Anomaly Detection             ║
║                   Library v0.1.6                   ║
║              Current Implementation                 ║
╚════════════════════════════════════════════════════╝
```

## Quick Start

### Dependencies
```toml
[dependencies]
anomaly-grid = "0.1.6"

# Included automatically:
nalgebra = "0.33.2"
ndarray = "0.16.1" 
rayon = "1.10.0"
```

### Basic Usage
```rust
use anomaly_grid::*;

fn main() -> Result<(), String> {
    // Basic usage with current safety limits
    let mut model = AdvancedTransitionModel::new(3); // Will be capped at 3 internally
    let sequence = vec!["A", "B", "C", "A", "B", "D"]
        .iter().map(|s| s.to_string()).collect();
    model.build_context_tree(&sequence)?;
    let anomalies = model.detect_advanced_anomalies(&sequence, 0.1);
    Ok(())
}
```

## Core Components

### 1. AdvancedTransitionModel

The main analysis engine with built-in safety limits.

```rust
pub struct AdvancedTransitionModel {
    pub contexts: HashMap<Vec<String>, ContextNode>,     // Limited to 500 contexts max
    pub max_order: usize,                                // Capped at 3 internally
    pub quantum_representation: Option<QuantumState>,    // Simplified implementation
    pub spectral_decomposition: Option<SpectralAnalysis>,// Fallback implementation
}
```

**Constructor:**
```rust
let model = AdvancedTransitionModel::new(max_order: usize);
```
- `max_order`: Requested context length (internally capped at 3)
- Effective range: 1-3 regardless of input
- Memory scales as `alphabet_size^effective_order`

### 2. Context Nodes

Each context stores transition statistics with automatic probability calculation.

```rust
pub struct ContextNode {
    pub counts: HashMap<String, usize>,        // Raw transition counts
    pub probabilities: HashMap<String, f64>,   // Auto-normalized probabilities
    pub information_content: f64,              // -log2(entropy)
    pub entropy: f64,                          // Shannon entropy
    pub kl_divergence: f64,                    // KL divergence from uniform
}
```

### 3. Anomaly Scores

Multi-dimensional anomaly assessment with current scoring methods.

```rust
pub struct AnomalyScore {
    pub state_sequence: Vec<String>,           // The analyzed window
    pub likelihood: f64,                       // Variable-order probability
    pub information_theoretic_score: f64,      // Entropy-based surprise
    pub spectral_anomaly_score: f64,          // Deviation from uniform (simplified)
    pub quantum_coherence_measure: f64,        // Basic coherence measure
    pub topological_signature: Vec<f64>,       // [components, cycles, clustering]
    pub confidence_interval: (f64, f64),      // Bayesian confidence bounds
}
```

## Training Process: build_context_tree()

The training phase with current safety implementations.

```rust
pub fn build_context_tree(&mut self, sequence: &[String]) -> Result<(), String>
```

### Current Safety Features:

1. **Sequence Length Limiting**:
   ```rust
   // Automatic truncation to prevent excessive computation
   let max_len = std::cmp::min(sequence.len(), 200);
   let limited_sequence = &sequence[..max_len];
   ```

2. **Order Capping**:
   ```rust
   // Internally limits max_order regardless of input
   for window_size in 1..=std::cmp::min(self.max_order, 3) {
   ```

3. **Context Limiting**:
   ```rust
   // Prevents memory exhaustion
   if self.contexts.len() > 500 {
       break; // Stop creating new contexts
   }
   ```

### What Actually Happens:

1. **Context Extraction**: For each window size from 1 to effective_max_order (3):
   ```rust
   // Example: sequence=["A","B","C","A","B"] (limited to first 200 elements)
   // Creates contexts:
   // ["A"] -> "B" (count: 2)
   // ["B"] -> "C" (count: 1), ["B"] -> end (count: 1)  
   // ["A","B"] -> "C" (count: 1)
   // Up to 500 total contexts maximum
   ```

2. **Probability Calculation**: Automatic normalization:
   ```rust
   probability = count / total_transitions_from_context
   ```

3. **Information Measures**: Standard calculations:
   ```rust
   entropy = -Σ(p * log2(p))
   information_content = -log2(entropy)
   kl_divergence = Σ(p * log2(p / uniform_p))
   ```

4. **Simplified Spectral Analysis**: 
   ```rust
   // Current fallback implementation
   let eigenvalues = DVector::from_element(n_states, Complex::new(1.0, 0.0));
   let stationary_dist = DVector::from_element(n_states, 1.0 / n_states as f64);
   // Fixed values: mixing_time = 10.0, spectral_gap = 0.5
   ```

5. **Basic Quantum Representation**:
   ```rust
   // Equal superposition with amplitude = 1/sqrt(n_states)
   // Capped at 50 states maximum
   ```

### Training Data Requirements:

- **Minimum sequence length**: 2 elements
- **Maximum sequence length**: 200 elements (hard limit)
- **Maximum contexts**: 500 per model
- **Effective max order**: 3 (regardless of input)

## Detection Process: detect_advanced_anomalies()

Analyzes test sequences using sliding window with current limitations.

```rust
pub fn detect_advanced_anomalies(&self, sequence: &[String], threshold: f64) -> Vec<AnomalyScore>
```

### Window Analysis:

The function slides a window of size `effective_max_order + 1` across the sequence:

```rust
// Example: effective_max_order=3, sequence=["A","B","C","D","E"]
// Windows analyzed:
// ["A","B","C","D"] -> scores this 4-element window
// ["B","C","D","E"] -> scores this 4-element window  
```

### Five Scoring Methods (Current Implementation):

#### 1. Likelihood Score
```rust
// Uses variable-order context matching
for context_len in 1..=max_context_len {
    let context = sequence[i-context_len..i];
    if let Some(prob) = get_transition_probability(context, next_state) {
        best_prob = max(best_prob, prob); // Raw probability
    }
}
likelihood *= best_prob;
```
**Lower values = more anomalous**

#### 2. Information Score
```rust
// Combines context entropy with transition surprise
let surprise = -log2(transition_probability);
let score = (total_surprise + context_entropy) / sequence_length;
```
**Higher values = more anomalous**

#### 3. Spectral Score (Simplified)
```rust
// Uses uniform stationary distribution as baseline
let deviation = abs(observed_frequency - uniform_probability);
```

#### 4. Quantum Coherence (Basic)
```rust
// Simplified coherence from density matrix trace
let coherence = 1.0 - trace(density_matrix) / n_states;
```

#### 5. Topological Signature
- **Connected components**: Number of unique contexts
- **Cycle detection**: Repeated subsequence patterns
- **Local clustering**: Context interconnectedness measure

## Practical Usage Examples

### Network Security Monitoring

```rust
fn detect_network_anomalies() -> Result<(), String> {
    // Normal traffic patterns (within 200 element limit)
    let normal_traffic: Vec<String> = vec![
        "TCP_SYN", "TCP_ACK", "HTTP_GET", "HTTP_200", "TCP_FIN",
        "TCP_SYN", "TCP_ACK", "HTTPS_POST", "HTTP_201", "TCP_FIN",
        "UDP_DNS", "UDP_RESPONSE", "TCP_SYN", "TCP_ACK", "HTTP_GET", "HTTP_200",
    ]
    .into_iter()
    .map(String::from)
    .collect();

    let mut model = AdvancedTransitionModel::new(2); // Good for network protocols
    model.build_context_tree(&normal_traffic)?;

    // Suspicious traffic for anomaly detection
    let suspicious_traffic: Vec<String> = vec![
        "TCP_SYN", "TCP_RST", "TCP_SYN", "TCP_RST", "TCP_SYN", "TCP_RST", // Port scan
        "UNKNOWN_PROTOCOL", "MALFORMED_PACKET", "BUFFER_OVERFLOW", // Attack signatures
    ]
    .into_iter()
    .map(String::from)
    .collect();

    let anomalies = model.detect_advanced_anomalies(&suspicious_traffic, 0.01);
    
    for anomaly in anomalies {
        if anomaly.likelihood < 1e-6 {
            println!("🚨 Network threat: {:?}", anomaly.state_sequence);
            println!("Risk: {:.2e}", 1.0 - anomaly.likelihood);
        }
    }

    Ok(())
}
```

### User Behavior Analysis

```rust
fn analyze_user_sessions() -> Result<(), String> {
    // Normal user behavior
    let normal_sessions: Vec<String> = vec![
        "LOGIN", "DASHBOARD", "PROFILE", "SETTINGS", "LOGOUT",
        "LOGIN", "SEARCH", "VIEW_ITEM", "ADD_CART", "CHECKOUT", "LOGOUT",
        "LOGIN", "MESSAGES", "COMPOSE", "SEND", "LOGOUT",
    ]
    .into_iter()
    .map(String::from)
    .collect();

    let mut model = AdvancedTransitionModel::new(3);
    model.build_context_tree(&normal_sessions)?;

    // Detect privilege escalation
    let suspicious_session: Vec<String> = vec![
        "LOGIN", "ADMIN_PANEL", "USER_LIST", "DELETE_USER", "DELETE_USER",
        "BULK_DOWNLOAD", "SENSITIVE_DATA_ACCESS",
    ]
    .into_iter()
    .map(String::from)
    .collect();

    let anomalies = model.detect_advanced_anomalies(&suspicious_session, 0.05);

    for anomaly in anomalies {
        let risk_score = 1.0 - anomaly.likelihood;
        if risk_score > 0.95 {
            println!("⚠️ Suspicious behavior: {:?}", anomaly.state_sequence);
            println!("Risk: {:.4}", risk_score);
        }
    }

    Ok(())
}
```

### Financial Fraud Detection

```rust
fn detect_transaction_fraud() -> Result<(), String> {
    // Normal transaction patterns (limited dataset)
    let normal_transactions: Vec<String> = vec![
        "AUTH", "PURCHASE", "CONFIRM", "SETTLEMENT",
        "AUTH", "ATM_WITHDRAWAL", "CONFIRM", "SETTLEMENT",
        "AUTH", "ONLINE_PAYMENT", "CONFIRM", "SETTLEMENT"
    ].into_iter().map(String::from).collect();

    let mut model = AdvancedTransitionModel::new(2); // Lower order for stability
    model.build_context_tree(&normal_transactions)?;

    // Test suspicious patterns
    let test_transactions: Vec<String> = vec![
        "AUTH", "LARGE_PURCHASE", "FOREIGN_COUNTRY", "CONFIRM",    // Unusual location
        "VELOCITY_ALERT", "AUTH", "AUTH", "AUTH", "AUTH",          // Rapid transactions
    ].into_iter().map(String::from).collect();

    let anomalies = model.detect_advanced_anomalies(&test_transactions, 0.001);

    // Calculate fraud scores
    for anomaly in anomalies {
        let fraud_score = (1.0 - anomaly.likelihood) * anomaly.information_theoretic_score;
        if fraud_score > 2.0 {
            println!("💳 Fraud alert: {:?}", anomaly.state_sequence);
            println!("Fraud score: {:.4}", fraud_score);
        }
    }

    Ok(())
}
```

### System Log Analysis

```rust
fn analyze_system_logs() -> Result<(), String> {
    // Normal system events (within limits)
    let normal_logs: Vec<String> = vec![
        "BOOT", "SERVICE_START", "AUTH_SUCCESS", "FILE_ACCESS", "NETWORK_CONNECT",
        "CRON_START", "BACKUP_BEGIN", "BACKUP_SUCCESS", "CRON_END",
        "HEALTH_CHECK", "MONITOR_CLEAR", "SERVICE_HEALTHY"
    ].into_iter().map(String::from).collect();

    let mut model = AdvancedTransitionModel::new(3);
    model.build_context_tree(&normal_logs)?;

    // Test with security incidents
    let test_logs: Vec<String> = vec![
        "UNAUTHORIZED_ACCESS", "PRIVILEGE_ESCALATION", "FILE_CORRUPTION",
        "SERVICE_CRASH", "SERVICE_CRASH", "SERVICE_CRASH",  // Repeated crashes
        "ROOTKIT_DETECTED", "MALWARE_POSITIVE"
    ].into_iter().map(String::from).collect();

    let anomalies = model.detect_advanced_anomalies(&test_logs, 0.01);

    for anomaly in anomalies {
        if anomaly.likelihood < 1e-8 {
            println!("🖥️ Critical system event: {:?}", anomaly.state_sequence);
            println!("Severity: {:.2e}", 1.0 / anomaly.likelihood);
        }
    }

    Ok(())
}
```

### DNA Sequence Analysis

```rust
fn analyze_genetic_sequences() -> Result<(), String> {
    // Normal gene patterns (simplified dataset)
    let normal_genes: Vec<String> = vec![
        "ATG", "CGA", "TTC", "AAG", "GCT", "TAA",  // Gene 1
        "ATG", "CCG", "ATC", "GGC", "TTC", "TAG",  // Gene 2
        "ATG", "GAA", "CTG", "TGC", "CAG", "TGA"   // Gene 3
    ].into_iter().map(String::from).collect();

    let mut model = AdvancedTransitionModel::new(3); // Good for codon analysis
    model.build_context_tree(&normal_genes)?;

    // Test with mutations
    let test_dna: Vec<String> = vec![
        "XTG", "CGA", "TTC", "AAG", "GCT", "TAA",  // Invalid nucleotide
        "ATG", "CGA", "TTC", "AAG", "GCT",         // Missing stop codon
        "NNN", "UUU", "QQQ"                        // Invalid sequence
    ].into_iter().map(String::from).collect();

    let mutations = model.detect_advanced_anomalies(&test_dna, 0.01);

    for mutation in mutations {
        let mutation_prob = 1.0 - mutation.likelihood;
        if mutation_prob > 0.9 {
            println!("🧬 Genetic anomaly: {:?}", mutation.state_sequence);
            println!("Mutation probability: {:.2e}", mutation_prob);
        }
    }

    Ok(())
}
```

## Batch Processing

For analyzing multiple sequences in parallel with current limits:

```rust
use anomaly_grid::batch_process_sequences;

fn batch_analysis() -> Result<(), String> {
    let sequences = vec![
        // Keep sequences small for current implementation
        vec!["GET", "200", "POST", "201"].iter().map(|s| s.to_string()).collect(),
        vec!["CONNECT", "SELECT", "INSERT", "COMMIT"].iter().map(|s| s.to_string()).collect(),
        vec!["SYN", "ACK", "DATA", "FIN"].iter().map(|s| s.to_string()).collect(),
    ];
    
    // Process all sequences in parallel (each limited to 200 elements, 500 contexts)
    let results = batch_process_sequences(&sequences, 2, 0.05); // Use lower order
    
    for (i, (original, anomalies)) in sequences.iter().zip(results.iter()).enumerate() {
        println!("Sequence {}: {:?}", i + 1, original);
        println!("Anomalies detected: {}", anomalies.len());
    }
    
    Ok(())
}
```

---Will keep adding more examples eventually---