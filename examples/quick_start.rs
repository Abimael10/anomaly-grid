use anomaly_grid::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create detector with maximum context order of 3
    let mut detector = AnomalyDetector::new(3)?;

    // Train on normal patterns
    let normal_sequence = vec!["A".to_string(), "B".to_string(), "C".to_string()];
    detector.train(&normal_sequence)?;

    // Detect anomalies in test sequence
    let test_sequence = vec!["A".to_string(), "X".to_string(), "Y".to_string()];
    let anomalies = detector.detect_anomalies(&test_sequence, 0.01)?;

    // Analyze results
    for anomaly in anomalies {
        if anomaly.likelihood < 1e-6 {
            println!("🚨 HIGH THREAT: {:?}", anomaly.sequence);
            println!("   Likelihood: {:.2e}", anomaly.likelihood);
            println!("   Anomaly Strength: {:.3}", anomaly.anomaly_strength);
            println!("   Information Score: {:.3}", anomaly.information_score);
        }
    }

    println!("Anomaly detection completed successfully!");
    Ok(())
}
