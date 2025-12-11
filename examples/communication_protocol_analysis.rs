//! Communication protocol demo (12 symbols)
//!
//! Generates realistic START/SYNC/DATA/ACK sessions, trains on normals, and
//! reports how many steganographic/flow anomalies were caught.

use anomaly_grid::*;
use rand::prelude::*;
use rand::SeedableRng;

const SYMBOLS: &[&str] = &[
    "START", "SYNC", "DATA", "ACK", "NACK", "RETRY", "PAUSE", "RESUME", "CHECK", "ERROR", "STOP",
    "IDLE",
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = StdRng::seed_from_u64(2024);

    let normal_sessions = build_normals(&mut rng, 900, 45..120);
    let attack_sessions = build_attacks(&mut rng, 180, 45..120);

    let mut detector = AnomalyDetector::new(3)?;
    detector.train_sequences(&normal_sessions)?;

    let threshold = 0.33;
    let mut caught = 0usize;
    for seq in &attack_sessions {
        if max_strength(&detector, seq)? >= threshold {
            caught += 1;
        }
    }

    println!("Communication Protocol");
    println!("======================");
    println!(
        "Training set: {} normal sessions (alphabet: {} symbols, avg len: {:.1})",
        normal_sessions.len(),
        SYMBOLS.len(),
        average_len(&normal_sessions)
    );
    println!(
        "Attack set: {} sessions | Caught: {} | Missed: {}",
        attack_sessions.len(),
        caught,
        attack_sessions.len().saturating_sub(caught)
    );

    println!("\nSample attacks:");
    for (i, seq) in attack_sessions.iter().take(5).enumerate() {
        let strength = max_strength(&detector, seq)?;
        let label = if strength >= threshold { "FLAGGED" } else { "missed" };
        println!(
            "  #{:<2} {:<8} | strength {:.3} | first 14 symbols {:?}",
            i + 1,
            label,
            strength,
            seq.iter().take(14).collect::<Vec<_>>()
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

fn average_len(seq: &[Vec<String>]) -> f64 {
    if seq.is_empty() {
        return 0.0;
    }
    let sum: usize = seq.iter().map(|s| s.len()).sum();
    sum as f64 / seq.len() as f64
}

fn build_normals(rng: &mut StdRng, count: usize, len_range: std::ops::Range<usize>) -> Vec<Vec<String>> {
    (0..count)
        .map(|_| generate_session(rng, len_range.clone()))
        .collect()
}

fn build_attacks(rng: &mut StdRng, count: usize, len_range: std::ops::Range<usize>) -> Vec<Vec<String>> {
    (0..count)
        .map(|_| {
            let mut seq = generate_session(rng, len_range.clone());
            introduce_attack(rng, &mut seq);
            seq
        })
        .collect()
}

fn generate_session(rng: &mut StdRng, len_range: std::ops::Range<usize>) -> Vec<String> {
    const FLOWS: &[&[&str]] = &[
        &["START", "SYNC", "DATA", "DATA", "ACK", "STOP"],
        &["START", "SYNC", "DATA", "CHECK", "ACK", "DATA", "STOP"],
        &["START", "SYNC", "DATA", "CHECK", "NACK", "RETRY", "DATA", "ACK", "STOP"],
        &["START", "SYNC", "DATA", "PAUSE", "RESUME", "DATA", "ACK", "STOP"],
        &["START", "SYNC", "DATA", "ERROR", "RETRY", "START", "SYNC", "DATA", "ACK", "STOP"],
    ];

    let target_len = rng.gen_range(len_range);
    let mut seq: Vec<String> = if rng.gen_bool(0.7) {
        FLOWS
            .choose(rng)
            .unwrap()
            .iter()
            .map(|s| s.to_string())
            .collect()
    } else {
        vec!["START".to_string(), "SYNC".to_string()]
    };

    while seq.len() < target_len {
        let next = next_symbol(rng, &seq);
        seq.push(next.to_string());
    }
    seq.truncate(target_len);
    seq
}

fn introduce_attack(rng: &mut StdRng, seq: &mut Vec<String>) {
    if seq.len() < 10 {
        return;
    }
    match rng.gen_range(0..4) {
        0 => {
            // Timing delays
            for _ in 0..rng.gen_range(1..3) {
                let pos = rng.gen_range(2..seq.len() - 2);
                seq.insert(pos, ["PAUSE", "IDLE"].choose(rng).unwrap().to_string());
            }
        }
        1 => {
            // Symbol substitution
            let pos = rng.gen_range(1..seq.len() - 1);
            let replacement = match seq[pos].as_str() {
                "ACK" => "NACK",
                "DATA" => "CHECK",
                "SYNC" => "START",
                "PAUSE" => "IDLE",
                _ => "ERROR",
            };
            seq[pos] = replacement.to_string();
        }
        2 => {
            // Pattern disruption
            let pos = rng.gen_range(2..seq.len() - 2);
            seq[pos] = "ERROR".to_string();
        }
        _ => {
            // Premature stop
            let pos = rng.gen_range(1..seq.len() - 1);
            seq[pos] = "STOP".to_string();
        }
    }
}

fn next_symbol<'a>(rng: &mut StdRng, seq: &'a [String]) -> &'a str {
    if seq.len() >= 3 {
        let k = (&seq[seq.len() - 3][..], &seq[seq.len() - 2][..], &seq[seq.len() - 1][..]);
        if let Some(opts) = match k {
            ("START", "SYNC", "DATA") => Some(&["DATA", "ACK"][..]),
            ("SYNC", "DATA", "DATA") => Some(&["DATA", "ACK", "CHECK"][..]),
            ("DATA", "DATA", "ACK") => Some(&["DATA", "STOP"][..]),
            ("DATA", "CHECK", "NACK") => Some(&["RETRY", "ERROR"][..]),
            ("PAUSE", "RESUME", "DATA") => Some(&["DATA", "ACK"][..]),
            _ => None,
        } {
            return opts.choose(rng).unwrap();
        }
    }

    if seq.len() >= 2 {
        let k = (&seq[seq.len() - 2][..], &seq[seq.len() - 1][..]);
        if let Some(opts) = match k {
            ("START", "SYNC") => Some(&["DATA", "CHECK"][..]),
            ("SYNC", "DATA") => Some(&["DATA", "ACK", "CHECK"][..]),
            ("DATA", "DATA") => Some(&["DATA", "ACK", "CHECK", "PAUSE"][..]),
            ("DATA", "ACK") => Some(&["DATA", "STOP", "PAUSE"][..]),
            ("CHECK", "ACK") => Some(&["DATA", "STOP"][..]),
            ("CHECK", "NACK") => Some(&["RETRY", "ERROR"][..]),
            ("NACK", "RETRY") => Some(&["DATA", "SYNC"][..]),
            ("PAUSE", "RESUME") => Some(&["DATA", "SYNC"][..]),
            ("ERROR", "RETRY") => Some(&["START", "SYNC"][..]),
            ("STOP", "IDLE") => Some(&["START", "IDLE"][..]),
            _ => None,
        } {
            return opts.choose(rng).unwrap();
        }
    }

    let last = &seq[seq.len() - 1][..];
    let opts = match last {
        "START" => &["SYNC", "DATA"][..],
        "STOP" => &["IDLE", "START"][..],
        "IDLE" => &["START", "IDLE"][..],
        _ => &["DATA", "ACK", "CHECK"][..],
    };
    opts.choose(rng).unwrap()
}
