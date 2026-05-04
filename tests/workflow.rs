//! End-to-end workflows in concrete domains.
//!
//! Each test trains on a known-normal corpus from a real-world domain
//! (network flows, transactions, IoT sensors, system logs) and verifies
//! that anomalous payloads produce non-empty score lists with valid
//! numerical properties.

#![allow(clippy::float_cmp)]

mod common;
use common::s;

use anomaly_grid::{batch_score, AnomalyDetector};

fn assert_well_formed(scores: &[anomaly_grid::AnomalyScore]) {
    for s in scores {
        assert!((0.0..=1.0).contains(&s.likelihood));
        assert!((0.0..=1.0).contains(&s.anomaly_strength));
        assert!(s.information_score >= 0.0);
        assert!(s.likelihood.is_finite());
        assert!(s.information_score.is_finite());
        assert!(s.anomaly_strength.is_finite());
    }
}

/// Network-traffic state machine: trained on TCP/HTTP normals, flags a
/// port-scan + exploit-attempt sequence.
#[test]
fn network_security_workflow() {
    let normals: Vec<String> = ["TCP_SYN", "TCP_ACK", "HTTP_GET", "HTTP_200", "TCP_FIN",
        "UDP_DNS", "UDP_RESPONSE", "HTTPS_CONNECT", "TLS_HANDSHAKE", "HTTP_POST", "HTTP_201"]
        .repeat(50)
        .iter()
        .map(|s| (*s).to_string())
        .collect();

    let mut detector = AnomalyDetector::new(3).expect("detector");
    detector.train(&normals).expect("train");

    let attack = s(&[
        "TCP_SYN", "TCP_RST", "TCP_SYN", "TCP_RST",
        "MALFORMED_PACKET", "EXPLOIT_ATTEMPT",
    ]);
    let anomalies = detector.detect_anomalies(&attack, 0.1).expect("detect");
    assert!(!anomalies.is_empty(), "attack patterns should be flagged");
    assert_well_formed(&anomalies);
}

/// Transaction fraud: AUTH→PURCHASE→CONFIRM→SETTLE normal trains
/// flag a velocity-attack burst.
#[test]
fn financial_fraud_workflow() {
    let normals: Vec<String> = ["AUTH", "PURCHASE", "CONFIRM", "SETTLE",
        "AUTH", "ATM_WITHDRAWAL", "CONFIRM",
        "AUTH", "TRANSFER", "CONFIRM", "SETTLE"]
        .repeat(30)
        .iter()
        .map(|s| (*s).to_string())
        .collect();

    let mut detector = AnomalyDetector::new(4).expect("detector");
    detector.train(&normals).expect("train");

    let fraud = s(&[
        "VELOCITY_ALERT", "AUTH", "AUTH", "AUTH",
        "LARGE_AMOUNT", "FOREIGN_COUNTRY",
    ]);
    let anomalies = detector.detect_anomalies(&fraud, 0.05).expect("detect");
    assert!(!anomalies.is_empty(), "fraud patterns should be flagged");
    assert_well_formed(&anomalies);
}

/// Industrial IoT: sensor-reading sequence flags an equipment failure.
#[test]
fn industrial_iot_workflow() {
    let normals: Vec<String> = ["TEMP_NORMAL", "PRESSURE_NORMAL", "VIBRATION_LOW",
        "TEMP_NORMAL", "PRESSURE_HIGH", "VIBRATION_LOW",
        "TEMP_HIGH", "PRESSURE_NORMAL", "VIBRATION_NORMAL"]
        .repeat(40)
        .iter()
        .map(|s| (*s).to_string())
        .collect();

    let mut detector = AnomalyDetector::new(3).expect("detector");
    detector.train(&normals).expect("train");

    let failure = s(&[
        "TEMP_CRITICAL", "PRESSURE_CRITICAL", "VIBRATION_CRITICAL",
        "ALARM_BEARING_FAILURE",
    ]);
    let anomalies = detector.detect_anomalies(&failure, 0.01).expect("detect");
    assert!(!anomalies.is_empty(), "equipment failure should be flagged");
    assert_well_formed(&anomalies);
}

/// System logs: trains on benign sessions, flags a security incident.
#[test]
fn system_log_workflow() {
    let normals: Vec<String> = ["USER_LOGIN", "FILE_ACCESS", "PROCESS_START", "USER_LOGOUT",
        "SYSTEM_BOOT", "SERVICE_START", "NETWORK_CONNECT", "SERVICE_STOP",
        "BACKUP_START", "BACKUP_SUCCESS", "CLEANUP_TEMP"]
        .repeat(25)
        .iter()
        .map(|s| (*s).to_string())
        .collect();

    let mut detector = AnomalyDetector::new(3).expect("detector");
    detector.train(&normals).expect("train");

    let incident = s(&[
        "FAILED_LOGIN", "FAILED_LOGIN", "FAILED_LOGIN",
        "PRIVILEGE_ESCALATION", "SUSPICIOUS_FILE_ACCESS", "DATA_EXFILTRATION",
    ]);
    let anomalies = detector.detect_anomalies(&incident, 0.01).expect("detect");
    assert!(!anomalies.is_empty(), "security incident should be flagged");
    assert_well_formed(&anomalies);
}

/// `batch_score` parallelises detection over many candidate sequences.
#[test]
fn batch_score_workflow() {
    let mut detector = AnomalyDetector::new(2).expect("detector");
    detector
        .train(&s(&["NORMAL", "PATTERN", "NORMAL", "PATTERN", "NORMAL", "PATTERN"]))
        .expect("train");

    let candidates = vec![
        s(&["NORMAL", "PATTERN", "A"]),
        s(&["NORMAL", "PATTERN", "B"]),
        s(&["ANOMALY", "PATTERN", "X"]),
        s(&["ANOMALY", "PATTERN", "Y"]),
    ];
    let results = batch_score(&detector, &candidates, 0.1).expect("batch_score");
    assert_eq!(results.len(), candidates.len());
    for set in &results {
        assert_well_formed(set);
    }
}
