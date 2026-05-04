//! Protein folding demo (20 amino acids)
//!
//! Synthesizes biologically plausible sequences, trains on normals, and reports
//! how many misfolded sequences were flagged (no precision jargon).

use anomaly_grid::*;
use rand::prelude::*;
use rand::SeedableRng;

const AA: &[&str] = &[
    "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y",
    "V",
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = StdRng::seed_from_u64(99);

    // Keep sizes moderate so the example finishes quickly while still being realistic.
    let normal_sequences = build_normals(&mut rng, 450, 120..210);
    let misfolded_sequences = build_misfolds(&mut rng, 120, 120..210);

    println!("Training detector on protein folds...");
    let mut detector = AnomalyDetector::new(4)?;
    detector.train_sequences(&normal_sequences)?;

    let threshold = 0.40;
    let mut caught = 0usize;
    for seq in &misfolded_sequences {
        if max_strength(&detector, seq)? >= threshold {
            caught += 1;
        }
    }

    println!("Protein Folding");
    println!("===============");
    println!(
        "Training set: {} normal sequences (alphabet: {} aa, avg len: {:.1})",
        normal_sequences.len(),
        AA.len(),
        average_len(&normal_sequences)
    );
    println!(
        "Misfolded set: {} sequences | Caught: {} | Missed: {}",
        misfolded_sequences.len(),
        caught,
        misfolded_sequences.len().saturating_sub(caught)
    );

    println!("\nSample misfolds:");
    for (i, seq) in misfolded_sequences.iter().take(5).enumerate() {
        let strength = max_strength(&detector, seq)?;
        let label = if strength >= threshold { "FLAGGED" } else { "missed" };
        println!(
            "  #{:<2} {:<8} | strength {:.3} | first 24 aa {}",
            i + 1,
            label,
            strength,
            seq.iter().take(24).cloned().collect::<String>()
        );
    }

    Ok(())
}

fn average_len(seq: &[Vec<String>]) -> f64 {
    if seq.is_empty() {
        return 0.0;
    }
    let sum: usize = seq.iter().map(|s| s.len()).sum();
    sum as f64 / seq.len() as f64
}

fn max_strength(detector: &AnomalyDetector, seq: &[String]) -> AnomalyGridResult<f64> {
    let anomalies = detector.detect_anomalies(seq, 0.0)?;
    Ok(anomalies
        .iter()
        .map(|a| a.anomaly_strength)
        .fold(0.0, f64::max))
}

fn build_normals(rng: &mut StdRng, count: usize, len_range: std::ops::Range<usize>) -> Vec<Vec<String>> {
    (0..count)
        .map(|_| generate_fold(rng, len_range.clone()))
        .collect()
}

fn build_misfolds(rng: &mut StdRng, count: usize, len_range: std::ops::Range<usize>) -> Vec<Vec<String>> {
    (0..count)
        .map(|_| {
            let mut seq = generate_fold(rng, len_range.clone());
            introduce_misfold(rng, &mut seq);
            seq
        })
        .collect()
}

fn generate_fold(rng: &mut StdRng, len_range: std::ops::Range<usize>) -> Vec<String> {
    const SIGNALS: &[&[&str]] = &[&["M", "K", "L", "L", "F"], &["L", "L", "A", "A", "A"]];
    const ALPHA: &[&str] = &["A", "E", "L", "M", "F", "I"];
    const BETA: &[&str] = &["I", "Y", "F", "V", "L"];
    const TURN: &[&str] = &["G", "N", "P", "S", "T"];
    const HYDROPHOBIC: &[&str] = &["A", "I", "L", "M", "F", "W", "Y", "V"];
    const HYDROPHILIC: &[&str] = &["R", "N", "D", "Q", "E", "H", "K", "S", "T"];

    let mut seq: Vec<String> = vec!["M".to_string()];
    if rng.gen_bool(0.22) {
        let signal = SIGNALS.choose(rng).unwrap();
        seq.extend(signal.iter().map(|s| s.to_string()));
    }

    let target_len = rng.gen_range(len_range);
    while seq.len() < target_len {
        let remaining = target_len - seq.len();
        match rng.gen_range(0..6) {
            0 if remaining >= 10 => {
                let len = rng.gen_range(8..15).min(remaining);
                for _ in 0..len {
                    seq.push(ALPHA.choose(rng).unwrap().to_string());
                }
            }
            1 if remaining >= 8 => {
                let len = rng.gen_range(6..13).min(remaining);
                for _ in 0..len {
                    seq.push(BETA.choose(rng).unwrap().to_string());
                }
            }
            2 if remaining >= 4 => {
                let len = rng.gen_range(3..7).min(remaining);
                for _ in 0..len {
                    seq.push(TURN.choose(rng).unwrap().to_string());
                }
            }
            3 if remaining >= 4 => {
                seq.extend(["C", "A", "A", "C"].iter().map(|s| s.to_string()));
            }
            _ => {
                let aa_set = if rng.gen_bool(0.5) { HYDROPHILIC } else { HYDROPHOBIC };
                seq.push(aa_set.choose(rng).unwrap().to_string());
            }
        }
    }

    seq.truncate(target_len);
    seq
}

fn introduce_misfold(rng: &mut StdRng, seq: &mut [String]) {
    if seq.len() < 30 {
        return;
    }
    match rng.gen_range(0..4) {
        0 => {
            // Hydrophobic exposure
            for _ in 0..rng.gen_range(2..5) {
                let pos = rng.gen_range(5..seq.len() - 5);
                seq[pos] = ["F", "W", "I", "L"].choose(rng).unwrap().to_string();
            }
        }
        1 => {
            // Break disulfide
            for residue in seq.iter_mut() {
                if residue == "C" {
                    *residue = ["S", "T", "A"].choose(rng).unwrap().to_string();
                    break;
                }
            }
        }
        2 => {
            // Charge cluster
            let start = rng.gen_range(4..seq.len().saturating_sub(4));
            for i in start..(start + 4).min(seq.len()) {
                seq[i] = ["R", "H", "K", "D", "E"].choose(rng).unwrap().to_string();
            }
        }
        _ => {
            // Proline kink
            let pos = rng.gen_range(6..seq.len() - 6);
            seq[pos] = "P".to_string();
        }
    }
}
