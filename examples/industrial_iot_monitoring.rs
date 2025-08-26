//! Industrial IoT Monitoring Example
//!
//! This example demonstrates predictive maintenance and anomaly detection
//! in industrial IoT environments using anomaly-grid for equipment monitoring,
//! failure prediction, and operational optimization.

use anomaly_grid::*;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🏭 Industrial IoT Monitoring with Anomaly Grid");
    println!("Predictive maintenance and equipment failure detection\n");

    // Simulate 1 year of normal industrial operations
    let normal_operations = generate_industrial_data(365); // 1 year
    println!(
        "Generated {} sensor readings (1 year)",
        normal_operations.len()
    );

    // Initialize anomaly detection for IoT sensor data
    let mut detector = AnomalyDetector::new(4)?; // Order 4 for sensor patterns

    // Train on normal operational data
    detector.train(&normal_operations)?;

    // Equipment monitoring scenarios
    println!("\n🔧 Equipment Health Monitoring");

    let equipment_scenarios = vec![
        ("Bearing Wear", generate_bearing_wear_pattern()),
        ("Vibration Anomaly", generate_vibration_anomaly()),
        ("Temperature Drift", generate_temperature_drift()),
        ("Pressure Fluctuation", generate_pressure_fluctuation()),
        ("Lubrication Failure", generate_lubrication_failure()),
        ("Motor Imbalance", generate_motor_imbalance()),
        ("Seal Degradation", generate_seal_degradation()),
    ];

    let mut maintenance_alerts = 0;
    let mut critical_failures_prevented = 0;

    for (equipment_issue, sensor_sequence) in equipment_scenarios {
        println!("\nMonitoring: {equipment_issue}");

        let detect_start = Instant::now();
        let anomalies = detector.detect_anomalies(&sensor_sequence, 0.005)?; // Sensitive for safety
        let detect_time = detect_start.elapsed();

        if !anomalies.is_empty() {
            maintenance_alerts += 1;

            let severity = calculate_severity(&anomalies);
            let time_to_failure = estimate_time_to_failure(&sensor_sequence);
            let maintenance_cost = estimate_maintenance_cost(equipment_issue);

            println!("  ⚠️ MAINTENANCE ALERT");
            println!("  📊 Anomalies detected: {}", anomalies.len());
            println!("  Severity: {severity}");
            println!("  Time to failure: {time_to_failure} hours");
            println!("  Estimated maintenance cost: ${maintenance_cost:.0}");
            println!("  Detection time: {detect_time:?}");

            if severity == "CRITICAL" {
                critical_failures_prevented += 1;
                println!("  🚨 IMMEDIATE SHUTDOWN RECOMMENDED");
            }

            generate_maintenance_recommendation(equipment_issue, &severity, time_to_failure);
        } else {
            println!("  ✅ Equipment operating normally");
        }
    }

    // Production line monitoring
    println!("\n🏭 Production Line Monitoring");
    let production_data = generate_production_line_data();

    let line_start = Instant::now();
    let line_anomalies = detector.detect_anomalies(&production_data, 0.01)?;
    let line_time = line_start.elapsed();

    if !line_anomalies.is_empty() {
        let efficiency_impact = calculate_efficiency_impact(&line_anomalies);
        println!("Production anomalies detected: {}", line_anomalies.len());
        println!("Efficiency impact: {efficiency_impact:.1}%");
        println!("Detection time: {line_time:?}");
    } else {
        println!("Production line operating normally");
    }

    // Energy consumption monitoring
    println!("\n⚡ Energy Consumption Monitoring");
    let energy_data = generate_energy_consumption_data();

    let energy_start = Instant::now();
    let energy_anomalies = detector.detect_anomalies(&energy_data, 0.02)?;
    let energy_time = energy_start.elapsed();

    if !energy_anomalies.is_empty() {
        let energy_waste = calculate_energy_waste(&energy_anomalies);
        println!("Energy anomalies detected: {}", energy_anomalies.len());
        println!("Estimated energy waste: {energy_waste:.0} kWh");
        println!("Cost impact: ${:.2}", energy_waste * 0.12); // $0.12/kWh
        println!("Detection time: {energy_time:?}");
    } else {
        println!("Energy consumption within normal parameters");
    }

    // Batch processing for multiple machines
    println!("\n📦 Multi-Machine Batch Processing");
    let machine_data = vec![
        generate_machine_data("CNC_001"),
        generate_machine_data("PRESS_002"),
        generate_machine_data("ROBOT_003"),
        generate_machine_data("CONVEYOR_004"),
        generate_machine_data("WELDER_005"),
    ];

    let batch_start = Instant::now();
    let config = AnomalyGridConfig::default().with_max_order(6)?;
    let batch_results = batch_process_sequences(&machine_data, &config, 0.01)?;
    let batch_time = batch_start.elapsed();

    println!(
        "Processed {} machines in {:?}",
        machine_data.len(),
        batch_time
    );
    for (i, results) in batch_results.iter().enumerate() {
        let machine_names = [
            "CNC_001",
            "PRESS_002",
            "ROBOT_003",
            "CONVEYOR_004",
            "WELDER_005",
        ];
        println!(
            "  {}: {} anomalies detected",
            machine_names[i],
            results.len()
        );
    }

    // Summary and ROI calculation
    println!("\n📊 Predictive Maintenance Summary");
    println!("Maintenance alerts generated: {maintenance_alerts}");
    println!("Critical failures prevented: {critical_failures_prevented}");

    let downtime_prevented = critical_failures_prevented * 8; // 8 hours per failure
    let cost_savings = downtime_prevented as f64 * 5000.0; // $5000/hour downtime cost
    let maintenance_costs = maintenance_alerts as f64 * 2000.0; // $2000 per maintenance
    let net_savings = cost_savings - maintenance_costs;

    println!("Downtime prevented: {downtime_prevented} hours");
    println!("Cost savings: ${cost_savings:.0}");
    println!("Maintenance costs: ${maintenance_costs:.0}");
    println!("Net savings: ${net_savings:.0}");
    println!("ROI: {:.1}%", (net_savings / maintenance_costs) * 100.0);

    Ok(())
}

fn generate_industrial_data(days: usize) -> Vec<String> {
    let mut data = Vec::new();
    let readings_per_day = 1440; // One reading per minute

    let normal_patterns = vec![
        // Temperature readings
        vec!["TEMP_NORMAL", "TEMP_STABLE", "TEMP_WITHIN_RANGE"],
        vec!["TEMP_STARTUP", "TEMP_RISING", "TEMP_NORMAL", "TEMP_STABLE"],
        // Pressure readings
        vec!["PRESSURE_NORMAL", "PRESSURE_STABLE", "PRESSURE_OPTIMAL"],
        vec!["PRESSURE_STARTUP", "PRESSURE_BUILDING", "PRESSURE_NORMAL"],
        // Vibration readings
        vec!["VIBRATION_LOW", "VIBRATION_NORMAL", "VIBRATION_STABLE"],
        vec![
            "VIBRATION_STARTUP",
            "VIBRATION_SETTLING",
            "VIBRATION_NORMAL",
        ],
        // Flow readings
        vec!["FLOW_NORMAL", "FLOW_STABLE", "FLOW_OPTIMAL"],
        vec!["FLOW_STARTUP", "FLOW_RAMPING", "FLOW_NORMAL"],
        // Power consumption
        vec!["POWER_NORMAL", "POWER_EFFICIENT", "POWER_STABLE"],
        vec!["POWER_STARTUP", "POWER_RAMPING", "POWER_NORMAL"],
        // Lubrication system
        vec!["LUBE_PRESSURE_OK", "LUBE_FLOW_NORMAL", "LUBE_TEMP_OK"],
        vec!["LUBE_CYCLE_START", "LUBE_DISPENSING", "LUBE_CYCLE_COMPLETE"],
    ];

    for _ in 0..days {
        for _ in 0..readings_per_day {
            let pattern = &normal_patterns[data.len() % normal_patterns.len()];
            data.extend(pattern.iter().map(|s| s.to_string()));
        }
    }

    data
}

fn generate_bearing_wear_pattern() -> Vec<String> {
    vec![
        // Early stage
        "VIBRATION_NORMAL",
        "TEMP_NORMAL",
        "SOUND_NORMAL",
        // Developing wear
        "VIBRATION_SLIGHT_INCREASE",
        "TEMP_SLIGHT_RISE",
        "SOUND_CHANGE",
        "VIBRATION_IRREGULAR",
        "TEMP_FLUCTUATION",
        "SOUND_GRINDING",
        // Advanced wear
        "VIBRATION_HIGH",
        "TEMP_HIGH",
        "SOUND_LOUD_GRINDING",
        "VIBRATION_SEVERE",
        "TEMP_CRITICAL",
        "SOUND_METAL_ON_METAL",
        // Imminent failure
        "VIBRATION_EXTREME",
        "TEMP_OVERHEATING",
        "SOUND_CATASTROPHIC",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

fn generate_vibration_anomaly() -> Vec<String> {
    vec![
        "VIBRATION_NORMAL",
        "VIBRATION_NORMAL",
        "VIBRATION_NORMAL",
        "VIBRATION_SPIKE",
        "VIBRATION_HIGH",
        "VIBRATION_OSCILLATING",
        "VIBRATION_HARMONIC",
        "VIBRATION_RESONANCE",
        "VIBRATION_UNSTABLE",
        "VIBRATION_CRITICAL",
        "VIBRATION_SHUTDOWN_REQUIRED",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

fn generate_temperature_drift() -> Vec<String> {
    vec![
        "TEMP_NORMAL",
        "TEMP_NORMAL",
        "TEMP_SLIGHT_RISE",
        "TEMP_GRADUAL_INCREASE",
        "TEMP_ABOVE_NORMAL",
        "TEMP_TRENDING_UP",
        "TEMP_HIGH",
        "TEMP_VERY_HIGH",
        "TEMP_CRITICAL",
        "TEMP_OVERHEATING",
        "TEMP_SHUTDOWN_THRESHOLD",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

fn generate_pressure_fluctuation() -> Vec<String> {
    vec![
        "PRESSURE_NORMAL",
        "PRESSURE_STABLE",
        "PRESSURE_SLIGHT_DROP",
        "PRESSURE_FLUCTUATING",
        "PRESSURE_UNSTABLE",
        "PRESSURE_CYCLING",
        "PRESSURE_LOW",
        "PRESSURE_VERY_LOW",
        "PRESSURE_CRITICAL_LOW",
        "PRESSURE_SYSTEM_FAILURE",
        "PRESSURE_EMERGENCY_STOP",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

fn generate_lubrication_failure() -> Vec<String> {
    vec![
        "LUBE_PRESSURE_OK",
        "LUBE_FLOW_NORMAL",
        "LUBE_TEMP_OK",
        "LUBE_PRESSURE_DROP",
        "LUBE_FLOW_REDUCED",
        "LUBE_TEMP_RISE",
        "LUBE_PRESSURE_LOW",
        "LUBE_FLOW_INSUFFICIENT",
        "LUBE_TEMP_HIGH",
        "LUBE_SYSTEM_FAILURE",
        "LUBE_STARVATION",
        "LUBE_EMERGENCY",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

fn generate_motor_imbalance() -> Vec<String> {
    vec![
        "MOTOR_BALANCED",
        "CURRENT_NORMAL",
        "TORQUE_STABLE",
        "MOTOR_SLIGHT_IMBALANCE",
        "CURRENT_FLUCTUATION",
        "TORQUE_VARIATION",
        "MOTOR_IMBALANCED",
        "CURRENT_SPIKES",
        "TORQUE_IRREGULAR",
        "MOTOR_SEVERE_IMBALANCE",
        "CURRENT_OVERLOAD",
        "TORQUE_UNSTABLE",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

fn generate_seal_degradation() -> Vec<String> {
    vec![
        "SEAL_INTACT",
        "PRESSURE_STABLE",
        "NO_LEAKAGE",
        "SEAL_WEAR_INITIAL",
        "PRESSURE_SLIGHT_DROP",
        "MINOR_SEEPAGE",
        "SEAL_DEGRADED",
        "PRESSURE_LOSS",
        "VISIBLE_LEAK",
        "SEAL_FAILURE",
        "PRESSURE_CRITICAL",
        "MAJOR_LEAK",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

fn generate_production_line_data() -> Vec<String> {
    vec![
        // Normal production
        "CONVEYOR_NORMAL",
        "ROBOT_CYCLE_COMPLETE",
        "QUALITY_PASS",
        "CONVEYOR_NORMAL",
        "ROBOT_CYCLE_COMPLETE",
        "QUALITY_PASS",
        // Anomaly
        "CONVEYOR_SLOW",
        "ROBOT_CYCLE_DELAYED",
        "QUALITY_FAIL",
        "CONVEYOR_JAM",
        "ROBOT_ERROR",
        "QUALITY_REJECT",
        "PRODUCTION_STOP",
        "MAINTENANCE_REQUIRED",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

fn generate_energy_consumption_data() -> Vec<String> {
    vec![
        "POWER_BASELINE",
        "EFFICIENCY_NORMAL",
        "CONSUMPTION_OPTIMAL",
        "POWER_BASELINE",
        "EFFICIENCY_NORMAL",
        "CONSUMPTION_OPTIMAL",
        // Energy waste
        "POWER_SPIKE",
        "EFFICIENCY_DROP",
        "CONSUMPTION_HIGH",
        "POWER_EXCESSIVE",
        "EFFICIENCY_POOR",
        "CONSUMPTION_WASTEFUL",
        "POWER_CRITICAL",
        "EFFICIENCY_FAILURE",
        "CONSUMPTION_EXTREME",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

fn generate_machine_data(machine_id: &str) -> Vec<String> {
    vec![
        format!("{}_STARTUP", machine_id),
        format!("{}_NORMAL_OPERATION", machine_id),
        format!("{}_CYCLE_COMPLETE", machine_id),
        format!("{}_NORMAL_OPERATION", machine_id),
        format!("{}_MAINTENANCE_DUE", machine_id),
    ]
}

fn calculate_severity(anomalies: &[AnomalyScore]) -> String {
    let max_strength = anomalies
        .iter()
        .map(|a| a.anomaly_strength)
        .fold(0.0, f64::max);

    if max_strength > 0.9 {
        "CRITICAL".to_string()
    } else if max_strength > 0.7 {
        "HIGH".to_string()
    } else if max_strength > 0.5 {
        "MEDIUM".to_string()
    } else {
        "LOW".to_string()
    }
}

fn estimate_time_to_failure(sequence: &[String]) -> u32 {
    // Estimate based on severity indicators
    let critical_indicators = sequence
        .iter()
        .filter(|s| s.contains("CRITICAL") || s.contains("SEVERE") || s.contains("EXTREME"))
        .count();

    if critical_indicators > 3 {
        2 // 2 hours
    } else if critical_indicators > 1 {
        24 // 24 hours
    } else {
        168 // 1 week
    }
}

fn estimate_maintenance_cost(equipment_issue: &str) -> f64 {
    match equipment_issue {
        "Bearing Wear" => 5000.0,
        "Vibration Anomaly" => 3000.0,
        "Temperature Drift" => 2000.0,
        "Pressure Fluctuation" => 4000.0,
        "Lubrication Failure" => 1500.0,
        "Motor Imbalance" => 8000.0,
        "Seal Degradation" => 2500.0,
        _ => 3000.0,
    }
}

fn calculate_efficiency_impact(anomalies: &[AnomalyScore]) -> f64 {
    let avg_strength =
        anomalies.iter().map(|a| a.anomaly_strength).sum::<f64>() / anomalies.len() as f64;

    avg_strength * 15.0 // Up to 15% efficiency impact
}

fn calculate_energy_waste(anomalies: &[AnomalyScore]) -> f64 {
    let total_strength = anomalies.iter().map(|a| a.anomaly_strength).sum::<f64>();

    total_strength * 100.0 // kWh wasted
}

fn generate_maintenance_recommendation(
    equipment_issue: &str,
    severity: &str,
    time_to_failure: u32,
) {
    println!("  📋 MAINTENANCE RECOMMENDATION");

    let action = match severity {
        "CRITICAL" => "IMMEDIATE SHUTDOWN AND REPAIR",
        "HIGH" => "SCHEDULE URGENT MAINTENANCE",
        "MEDIUM" => "PLAN MAINTENANCE WITHIN 48 HOURS",
        _ => "MONITOR AND SCHEDULE ROUTINE MAINTENANCE",
    };

    println!("    Issue: {equipment_issue}");
    println!("    Action: {action}");
    println!("    Timeline: {time_to_failure} hours");

    let parts_needed = match equipment_issue {
        "Bearing Wear" => "Replacement bearings, lubricant",
        "Vibration Anomaly" => "Balancing weights, alignment tools",
        "Temperature Drift" => "Cooling system components, sensors",
        "Pressure Fluctuation" => "Seals, pressure regulators",
        "Lubrication Failure" => "Lubricant, filters, pumps",
        "Motor Imbalance" => "Motor components, alignment equipment",
        "Seal Degradation" => "Replacement seals, gaskets",
        _ => "Standard maintenance kit",
    };

    println!("    Parts needed: {parts_needed}");
}
