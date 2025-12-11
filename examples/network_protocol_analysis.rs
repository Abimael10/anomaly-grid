//! Network protocol state machine demo (16 states)
//!
//! Generates a large set of compliant TCP-like sessions, trains on them, and
//! then scans a separate anomaly set. Reports how many anomalous sessions were
//! caught (no precision/recall jargon).

use anomaly_grid::*;
use rand::prelude::*;
use rand::SeedableRng;

const STATES: &[&str] = &[
    "INIT",
    "LISTEN",
    "SYN_SENT",
    "SYN_RECV",
    "ESTABLISHED",
    "AUTH",
    "DATA_XFER",
    "FIN_WAIT1",
    "FIN_WAIT2",
    "CLOSE_WAIT",
    "CLOSING",
    "LAST_ACK",
    "TIME_WAIT",
    "CLOSED",
    "ERROR",
    "RESET",
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = StdRng::seed_from_u64(42);

    // Build a realistic training set (only normals) and a separate anomaly set.
    let normal_flows = build_normal_flows(&mut rng, 900, 60..140);
    let anomaly_flows = build_anomaly_flows(&mut rng, 140, 60..140);

    let mut detector = AnomalyDetector::new(4)?;
    detector.train_sequences(&normal_flows)?;

    // Score anomalies with a fixed threshold; anything above is flagged.
    let threshold = 0.38;
    let mut caught = 0usize;
    for seq in &anomaly_flows {
        if max_strength(&detector, seq)? >= threshold {
            caught += 1;
        }
    }

    println!("Network Protocol State Machine");
    println!("==============================");
    println!("Alphabet size: {} states", STATES.len());
    println!(
        "Training set: {} normal flows (avg len: {:.1})",
        normal_flows.len(),
        average_len(&normal_flows)
    );
    println!(
        "Anomaly set: {} flows | Caught: {} | Missed: {}",
        anomaly_flows.len(),
        caught,
        anomaly_flows.len().saturating_sub(caught)
    );

    // Show a few anomaly cases and whether they were flagged.
    println!("\nSample anomaly cases:");
    for (i, seq) in anomaly_flows.iter().take(5).enumerate() {
        let strength = max_strength(&detector, seq)?;
        let label = if strength >= threshold { "FLAGGED" } else { "missed" };
        println!(
            "  #{:<2} {:<8} | strength {:.3} | first 12 states {:?}",
            i + 1,
            label,
            strength,
            seq.iter().take(12).collect::<Vec<_>>()
        );
    }

    Ok(())
}

fn max_strength(detector: &AnomalyDetector, seq: &[String]) -> AnomalyGridResult<f64> {
    let anomalies = detector.detect_anomalies(seq, 0.0)?;
    Ok(anomalies
        .iter()
        .map(|a| a.anomaly_strength)
        .fold(0.0, f64::max))
}

fn average_len(flows: &[Vec<String>]) -> f64 {
    if flows.is_empty() {
        return 0.0;
    }
    let sum: usize = flows.iter().map(|f| f.len()).sum();
    sum as f64 / flows.len() as f64
}

fn build_normal_flows(rng: &mut StdRng, count: usize, len_range: std::ops::Range<usize>) -> Vec<Vec<String>> {
    (0..count)
        .map(|_| generate_normal_flow(rng, len_range.clone()))
        .collect()
}

fn build_anomaly_flows(rng: &mut StdRng, count: usize, len_range: std::ops::Range<usize>) -> Vec<Vec<String>> {
    (0..count)
        .map(|_| generate_anomalous_flow(rng, len_range.clone()))
        .collect()
}

fn generate_normal_flow(rng: &mut StdRng, len_range: std::ops::Range<usize>) -> Vec<String> {
    let target_len = rng.gen_range(len_range);
    let mut seq = vec!["INIT".to_string()];
    while seq.len() < target_len {
        let next = next_state(rng, &seq);
        seq.push(next.to_string());
    }
    seq.truncate(target_len);
    seq
}

fn generate_anomalous_flow(rng: &mut StdRng, len_range: std::ops::Range<usize>) -> Vec<String> {
    let mut seq = generate_normal_flow(rng, len_range);
    if seq.len() < 6 {
        return seq;
    }

    match rng.gen_range(0..4) {
        0 => {
            // Reset during data phase
            let pos = rng.gen_range(3..seq.len() - 2);
            seq[pos] = "RESET".to_string();
        }
        1 => {
            // Skip handshake
            seq.retain(|s| s != "LISTEN" && s != "SYN_SENT");
        }
        2 => {
            // Impossible jump
            let pos = rng.gen_range(2..seq.len() - 2);
            seq[pos] = "CLOSED".to_string();
        }
        _ => {
            // Loop in FIN_WAIT2
            let pos = rng.gen_range(4..seq.len());
            seq.insert(pos, "FIN_WAIT2".to_string());
        }
    }

    seq
}

fn next_state<'a>(rng: &mut StdRng, seq: &'a [String]) -> &'a str {
    // Order-3 preference
    if seq.len() >= 3 {
        let k = (&seq[seq.len() - 3][..], &seq[seq.len() - 2][..], &seq[seq.len() - 1][..]);
        if let Some(opts) = match k {
            ("INIT", "SYN_SENT", "SYN_RECV") => Some(&["ESTABLISHED"][..]),
            ("SYN_RECV", "ESTABLISHED", "AUTH") => Some(&["DATA_XFER"][..]),
            ("ESTABLISHED", "AUTH", "DATA_XFER") => Some(&["DATA_XFER", "FIN_WAIT1"][..]),
            ("DATA_XFER", "FIN_WAIT1", "FIN_WAIT2") => Some(&["TIME_WAIT"][..]),
            ("FIN_WAIT2", "TIME_WAIT", "CLOSED") => Some(&["INIT"][..]),
            ("TIME_WAIT", "CLOSED", "INIT") => Some(&["LISTEN", "SYN_SENT"][..]),
            _ => None,
        } {
            return opts.choose(rng).unwrap();
        }
    }

    // Order-2 next-best
    if seq.len() >= 2 {
        let k = (&seq[seq.len() - 2][..], &seq[seq.len() - 1][..]);
        if let Some(opts) = match k {
            ("INIT", "LISTEN") => Some(&["SYN_RECV", "CLOSED"][..]),
            ("INIT", "SYN_SENT") => Some(&["SYN_RECV", "ESTABLISHED"][..]),
            ("SYN_RECV", "ESTABLISHED") => Some(&["AUTH", "DATA_XFER"][..]),
            ("ESTABLISHED", "AUTH") => Some(&["DATA_XFER", "ESTABLISHED"][..]),
            ("AUTH", "DATA_XFER") => Some(&["DATA_XFER", "FIN_WAIT1"][..]),
            ("DATA_XFER", "DATA_XFER") => Some(&["DATA_XFER", "FIN_WAIT1", "CLOSE_WAIT"][..]),
            ("FIN_WAIT1", "FIN_WAIT2") => Some(&["TIME_WAIT", "CLOSED"][..]),
            ("TIME_WAIT", "CLOSED") => Some(&["INIT", "LISTEN"][..]),
            ("CLOSED", "INIT") => Some(&["LISTEN", "SYN_SENT"][..]),
            ("ERROR", "RESET") => Some(&["INIT", "CLOSED"][..]),
            _ => None,
        } {
            return opts.choose(rng).unwrap();
        }
    }

    // Order-1 fallback
    let last = &seq[seq.len() - 1][..];
    let opts = match last {
        "INIT" => &["LISTEN", "SYN_SENT"][..],
        "LISTEN" => &["SYN_RECV", "CLOSED"][..],
        "SYN_SENT" => &["SYN_RECV", "ESTABLISHED", "CLOSED"][..],
        "SYN_RECV" => &["ESTABLISHED", "FIN_WAIT1", "RESET"][..],
        "ESTABLISHED" => &["AUTH", "DATA_XFER", "FIN_WAIT1", "CLOSE_WAIT"][..],
        "AUTH" => &["DATA_XFER", "ERROR", "ESTABLISHED"][..],
        "DATA_XFER" => &["DATA_XFER", "FIN_WAIT1", "CLOSE_WAIT", "ESTABLISHED"][..],
        "FIN_WAIT1" => &["FIN_WAIT2", "CLOSING", "TIME_WAIT"][..],
        "FIN_WAIT2" => &["TIME_WAIT", "CLOSED"][..],
        "CLOSE_WAIT" => &["LAST_ACK", "CLOSED"][..],
        "CLOSING" => &["TIME_WAIT", "CLOSED"][..],
        "LAST_ACK" => &["CLOSED", "TIME_WAIT"][..],
        "TIME_WAIT" => &["CLOSED", "INIT"][..],
        "CLOSED" => &["INIT", "LISTEN"][..],
        "ERROR" => &["RESET", "CLOSED", "INIT"][..],
        "RESET" => &["INIT", "CLOSED"][..],
        _ => &["INIT"][..],
    };
    opts.choose(rng).unwrap()
}
