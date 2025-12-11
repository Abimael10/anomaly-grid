use anomaly_grid::*;

#[test]
fn train_sequences_preserves_vocabulary_across_batches() -> Result<(), Box<dyn std::error::Error>> {
    let sequences: Vec<Vec<String>> = vec![vec!["A", "B", "A"], vec!["B", "C", "B"]]
        .into_iter()
        .map(|seq| seq.into_iter().map(String::from).collect())
        .collect();

    let mut detector = AnomalyDetector::new(2)?;
    detector.train_sequences(&sequences)?;

    let model = detector.model();
    assert!(model.state_mapping().contains_key("A"));
    assert!(model.state_mapping().contains_key("C"));

    // Earlier states should stay known when later sequences are trained
    let prob = model.get_best_context_probability(&["A".to_string()], "B");
    assert!(prob > 0.0);

    Ok(())
}

#[test]
fn marginal_probability_stable_across_orders() -> Result<(), Box<dyn std::error::Error>> {
    let sequence: Vec<String> = vec!["A", "B", "A", "A"]
        .into_iter()
        .map(String::from)
        .collect();

    let mut order_one = MarkovModel::new(1)?;
    order_one.train(&sequence)?;

    let mut order_three = MarkovModel::new(3)?;
    order_three.train(&sequence)?;

    let p1 = order_one.get_marginal_probability("A");
    let p3 = order_three.get_marginal_probability("A");

    assert!(
        (p1 - p3).abs() < 1e-9,
        "marginals should not depend on max_order"
    );
    Ok(())
}

#[test]
fn information_score_prefers_longer_supported_contexts() -> Result<(), Box<dyn std::error::Error>> {
    // B->C is common overall, but AB->C is unseen, so the length-2 context should dominate scoring
    let training: Vec<Vec<String>> = vec![
        vec!["A", "B", "D"],
        vec!["A", "B", "D"],
        vec!["B", "C"],
        vec!["X", "B", "C"],
        vec!["Y", "B", "C"],
    ]
    .into_iter()
    .map(|seq| seq.into_iter().map(String::from).collect())
    .collect();

    let mut detector = AnomalyDetector::new(2)?;
    detector.train_sequences(&training)?;

    let test_seq: Vec<String> = vec!["A", "B", "C"].into_iter().map(String::from).collect();
    let anomalies = detector.detect_anomalies(&test_seq, 0.0)?;

    assert_eq!(anomalies.len(), 1);
    let model = detector.model();
    let p_b_given_a = model.get_best_context_probability(&["A".to_string()], "B");
    let p_c_given_ab = model.get_best_context_probability(&["A".to_string(), "B".to_string()], "C");
    let expected_info = (-p_b_given_a.log2() - p_c_given_ab.log2()) / 2.0;

    assert!(
        (anomalies[0].information_score - expected_info).abs() < 1e-9,
        "information score should use longest-supported context; expected ~{expected_info}"
    );

    Ok(())
}
