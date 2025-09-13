//! Music Pattern Analysis
//!
//! This example demonstrates using anomaly-grid for analyzing musical note sequences
//! to understand composition patterns and detect stylistic anomalies.
//!
//! ## Use Case:
//! - Musical notes/chords are naturally categorical (finite alphabet)
//! - Musical patterns ARE the note sequences
//! - No missing fundamental features - note sequences contain all needed musical information
//! - Mathematical approach (Markov chains) is perfectly suited for music analysis
//! - Limitations are purely computational (scale/performance), not missing features
//!
//! ## Value Provided:
//! - Musical style analysis and classification
//! - Composition pattern recognition
//! - Music recommendation insights
//! - Educational music theory analysis
//!
//! ## Computational Limitations (Not Feature Gaps):
//! - Scale: Processing large music databases
//! - Performance: Real-time analysis of music streams
//! - Memory: Large musical vocabularies and pattern libraries
//! - Distribution: Analysis across multiple music collections

#![allow(clippy::uninlined_format_args)]

use anomaly_grid::*;
use std::time::Instant;
use std::collections::HashMap;

/// Musical elements that can be analyzed
#[derive(Debug, Clone, PartialEq)]
pub enum MusicalElement {
    // Notes
    C,
    CSharp,
    D,
    DSharp,
    E,
    F,
    FSharp,
    G,
    GSharp,
    A,
    ASharp,
    B,
    
    // Chords
    CMajor,
    CMinor,
    DMajor,
    DMinor,
    EMajor,
    EMinor,
    FMajor,
    FMinor,
    GMajor,
    GMinor,
    AMajor,
    AMinor,
    BMajor,
    BMinor,
    
    // Musical structures
    Verse,
    Chorus,
    Bridge,
    Intro,
    Outro,
    Solo,
    
    // Rhythmic elements
    Whole,
    Half,
    Quarter,
    Eighth,
    Sixteenth,
    
    // Rests
    Rest,
    LongRest,
}

impl MusicalElement {
    fn to_string(&self) -> String {
        match self {
            MusicalElement::C => "C".to_string(),
            MusicalElement::CSharp => "C_SHARP".to_string(),
            MusicalElement::D => "D".to_string(),
            MusicalElement::DSharp => "D_SHARP".to_string(),
            MusicalElement::E => "E".to_string(),
            MusicalElement::F => "F".to_string(),
            MusicalElement::FSharp => "F_SHARP".to_string(),
            MusicalElement::G => "G".to_string(),
            MusicalElement::GSharp => "G_SHARP".to_string(),
            MusicalElement::A => "A".to_string(),
            MusicalElement::ASharp => "A_SHARP".to_string(),
            MusicalElement::B => "B".to_string(),
            MusicalElement::CMajor => "C_MAJOR".to_string(),
            MusicalElement::CMinor => "C_MINOR".to_string(),
            MusicalElement::DMajor => "D_MAJOR".to_string(),
            MusicalElement::DMinor => "D_MINOR".to_string(),
            MusicalElement::EMajor => "E_MAJOR".to_string(),
            MusicalElement::EMinor => "E_MINOR".to_string(),
            MusicalElement::FMajor => "F_MAJOR".to_string(),
            MusicalElement::FMinor => "F_MINOR".to_string(),
            MusicalElement::GMajor => "G_MAJOR".to_string(),
            MusicalElement::GMinor => "G_MINOR".to_string(),
            MusicalElement::AMajor => "A_MAJOR".to_string(),
            MusicalElement::AMinor => "A_MINOR".to_string(),
            MusicalElement::BMajor => "B_MAJOR".to_string(),
            MusicalElement::BMinor => "B_MINOR".to_string(),
            MusicalElement::Verse => "VERSE".to_string(),
            MusicalElement::Chorus => "CHORUS".to_string(),
            MusicalElement::Bridge => "BRIDGE".to_string(),
            MusicalElement::Intro => "INTRO".to_string(),
            MusicalElement::Outro => "OUTRO".to_string(),
            MusicalElement::Solo => "SOLO".to_string(),
            MusicalElement::Whole => "WHOLE".to_string(),
            MusicalElement::Half => "HALF".to_string(),
            MusicalElement::Quarter => "QUARTER".to_string(),
            MusicalElement::Eighth => "EIGHTH".to_string(),
            MusicalElement::Sixteenth => "SIXTEENTH".to_string(),
            MusicalElement::Rest => "REST".to_string(),
            MusicalElement::LongRest => "LONG_REST".to_string(),
        }
    }
}

/// Analysis result for a musical composition
#[derive(Debug, Clone)]
pub struct MusicalAnalysis {
    pub composition_id: String,
    pub musical_sequence: Vec<String>,
    pub creativity_score: f64,
    pub musical_patterns: Vec<String>,
    pub style_insights: Vec<String>,
    pub complexity_level: ComplexityLevel,
    pub explanation: String,
    pub similar_compositions_found: usize,
}

/// Musical complexity levels
#[derive(Debug, Clone)]
pub enum ComplexityLevel {
    Simple,
    Moderate,
    Complex,
    Experimental,
}

impl ComplexityLevel {
    fn from_score(score: f64) -> Self {
        if score >= 0.8 {
            ComplexityLevel::Experimental
        } else if score >= 0.6 {
            ComplexityLevel::Complex
        } else if score >= 0.3 {
            ComplexityLevel::Moderate
        } else {
            ComplexityLevel::Simple
        }
    }
    
    fn to_string(&self) -> &str {
        match self {
            ComplexityLevel::Simple => "SIMPLE",
            ComplexityLevel::Moderate => "MODERATE",
            ComplexityLevel::Complex => "COMPLEX",
            ComplexityLevel::Experimental => "EXPERIMENTAL",
        }
    }
}

/// Musical pattern analyzer
pub struct MusicPatternAnalyzer {
    detector: AnomalyDetector,
    composition_patterns: HashMap<String, Vec<String>>,
    analysis_results: Vec<MusicalAnalysis>,
    performance_metrics: HashMap<String, f64>,
}

impl MusicPatternAnalyzer {
    /// Create new music pattern analyzer
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let detector = AnomalyDetector::new(4)?; // 4th order for complex musical patterns
        
        Ok(Self {
            detector,
            composition_patterns: HashMap::new(),
            analysis_results: Vec::new(),
            performance_metrics: HashMap::new(),
        })
    }
    
    /// Train on common musical patterns
    pub fn train_on_common_patterns(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("🔍 Training music analyzer on common musical patterns...");
        
        let common_patterns = self.generate_common_musical_patterns();
        let start_time = Instant::now();
        
        for pattern in &common_patterns {
            self.detector.train(pattern)?;
        }
        
        let training_time = start_time.elapsed();
        self.performance_metrics.insert("training_time_ms".to_string(), training_time.as_millis() as f64);
        
        println!("✅ Training completed in {:?}", training_time);
        println!("📊 Trained on {} common musical patterns", common_patterns.len());
        
        Ok(())
    }
    
    /// Analyze a musical composition
    pub fn analyze_composition(&mut self, elements: &[MusicalElement], composition_id: &str, threshold: f64) 
        -> Result<Option<MusicalAnalysis>, Box<dyn std::error::Error>> {
        
        let musical_sequence: Vec<String> = elements.iter()
            .map(|e| e.to_string())
            .collect();
        
        // Store composition pattern
        self.composition_patterns.insert(composition_id.to_string(), musical_sequence.clone());
        
        let detection_start = Instant::now();
        let anomalies = self.detector.detect_anomalies(&musical_sequence, threshold)?;
        let detection_time = detection_start.elapsed();
        
        self.performance_metrics.insert("last_detection_time_ms".to_string(), detection_time.as_millis() as f64);
        
        if anomalies.is_empty() {
            // Even common patterns get analyzed for insights
            let analysis = self.create_common_pattern_analysis(composition_id, &musical_sequence)?;
            return Ok(Some(analysis));
        }
        
        // Find the highest scoring anomaly
        let max_anomaly = anomalies.iter()
            .max_by(|a, b| a.anomaly_strength.partial_cmp(&b.anomaly_strength).unwrap())
            .unwrap();
        
        // Identify musical patterns
        let musical_patterns = self.identify_musical_patterns(&musical_sequence, max_anomaly);
        
        // Generate style insights
        let style_insights = self.generate_style_insights(&musical_sequence, max_anomaly);
        
        // Count similar compositions for context
        let similar_compositions = self.count_similar_compositions(&musical_sequence);
        
        let complexity = ComplexityLevel::from_score(max_anomaly.anomaly_strength);
        
        let explanation = format!(
            "Musical analysis: Anomaly strength: {:.3}, Likelihood: {:.6}, Information content: {:.3}. This musical sequence deviates from common patterns with {:.1}% confidence. Creative and stylistic insights identified.",
            max_anomaly.anomaly_strength,
            max_anomaly.likelihood,
            max_anomaly.information_score,
            (1.0 - max_anomaly.likelihood) * 100.0
        );
        
        let analysis = MusicalAnalysis {
            composition_id: composition_id.to_string(),
            musical_sequence,
            creativity_score: max_anomaly.anomaly_strength,
            musical_patterns,
            style_insights,
            complexity_level: complexity,
            explanation,
            similar_compositions_found: similar_compositions,
        };
        
        self.analysis_results.push(analysis.clone());
        
        Ok(Some(analysis))
    }
    
    /// Create analysis for common patterns
    fn create_common_pattern_analysis(&mut self, composition_id: &str, musical_sequence: &[String]) 
        -> Result<MusicalAnalysis, Box<dyn std::error::Error>> {
        
        let musical_patterns = vec!["COMMON_MUSICAL_PATTERN".to_string()];
        let style_insights = self.generate_style_insights(musical_sequence, &AnomalyScore {
            sequence: musical_sequence.to_vec(),
            likelihood: 0.9,
            log_likelihood: 0.9_f64.ln(),
            information_score: 1.0,
            anomaly_strength: 0.1,
        });
        
        let analysis = MusicalAnalysis {
            composition_id: composition_id.to_string(),
            musical_sequence: musical_sequence.to_vec(),
            creativity_score: 0.1,
            musical_patterns,
            style_insights,
            complexity_level: ComplexityLevel::Simple,
            explanation: "Musical composition follows common patterns and conventions.".to_string(),
            similar_compositions_found: self.count_similar_compositions(musical_sequence),
        };
        
        self.analysis_results.push(analysis.clone());
        Ok(analysis)
    }
    
    /// Identify specific musical patterns
    fn identify_musical_patterns(&self, musical_sequence: &[String], anomaly: &AnomalyScore) -> Vec<String> {
        let mut patterns = Vec::new();
        
        // Check for unusual chord progressions
        if self.detect_unusual_chord_progression(musical_sequence) {
            patterns.push("UNUSUAL_CHORD_PROGRESSION".to_string());
        }
        
        // Check for experimental structure
        if self.detect_experimental_structure(musical_sequence) {
            patterns.push("EXPERIMENTAL_STRUCTURE".to_string());
        }
        
        // Check for complex rhythmic patterns
        if self.detect_complex_rhythm(musical_sequence) {
            patterns.push("COMPLEX_RHYTHMIC_PATTERN".to_string());
        }
        
        // Check for modal or atonal elements
        if self.detect_modal_elements(musical_sequence) {
            patterns.push("MODAL_OR_ATONAL_ELEMENTS".to_string());
        }
        
        // Check for repetitive patterns
        if self.detect_repetitive_patterns(musical_sequence) {
            patterns.push("HIGHLY_REPETITIVE_PATTERN".to_string());
        }
        
        // High information content suggests very creative composition
        if anomaly.information_score > 4.0 {
            patterns.push("HIGHLY_CREATIVE_COMPOSITION".to_string());
        }
        
        patterns
    }
    
    /// Generate style insights
    fn generate_style_insights(&self, musical_sequence: &[String], anomaly: &AnomalyScore) -> Vec<String> {
        let mut insights = Vec::new();
        
        // Analyze chord usage
        let major_chords = musical_sequence.iter().filter(|e| e.contains("MAJOR")).count();
        let minor_chords = musical_sequence.iter().filter(|e| e.contains("MINOR")).count();
        
        if major_chords > minor_chords * 2 {
            insights.push("Predominantly major tonality - bright, uplifting character".to_string());
        } else if minor_chords > major_chords {
            insights.push("Minor tonality emphasis - melancholic or dramatic character".to_string());
        }
        
        // Analyze structural elements
        let verse_count = musical_sequence.iter().filter(|e| e.as_str() == "VERSE").count();
        let chorus_count = musical_sequence.iter().filter(|e| e.as_str() == "CHORUS").count();
        
        if chorus_count > verse_count {
            insights.push("Chorus-heavy structure - emphasis on memorable hooks".to_string());
        }
        
        // Analyze rhythmic complexity
        let complex_rhythms = musical_sequence.iter()
            .filter(|e| e.as_str() == "SIXTEENTH" || e.as_str() == "EIGHTH")
            .count();
        if complex_rhythms > musical_sequence.len() / 3 {
            insights.push("Complex rhythmic patterns - sophisticated timing".to_string());
        }
        
        // Analyze rests and space
        let rest_count = musical_sequence.iter().filter(|e| e.contains("REST")).count();
        if rest_count > musical_sequence.len() / 4 {
            insights.push("Effective use of silence and space in composition".to_string());
        }
        
        // Creativity insights
        if anomaly.anomaly_strength < 0.2 {
            insights.push("Traditional composition following established patterns".to_string());
        } else if anomaly.anomaly_strength > 0.6 {
            insights.push("Highly creative composition with innovative elements".to_string());
        }
        
        insights
    }
    
    /// Detect unusual chord progressions
    fn detect_unusual_chord_progression(&self, sequence: &[String]) -> bool {
        // Look for non-standard chord progressions
        let chords: Vec<&String> = sequence.iter().filter(|e| e.contains("MAJOR") || e.contains("MINOR")).collect();
        
        if chords.len() >= 3 {
            // Check for unusual progressions (simplified)
            for i in 0..chords.len().saturating_sub(2) {
                // Look for tritone substitutions or other unusual progressions
                if chords[i].contains("C_") && chords[i+1].contains("F_SHARP") {
                    return true;
                }
            }
        }
        false
    }
    
    /// Detect experimental structure
    fn detect_experimental_structure(&self, sequence: &[String]) -> bool {
        let structure_elements = sequence.iter()
            .filter(|e| e.as_str() == "VERSE" || e.as_str() == "CHORUS" || e.as_str() == "BRIDGE" || e.as_str() == "SOLO")
            .count();
        
        // Unusual if very few or very many structural elements
        structure_elements == 0 || structure_elements > sequence.len() / 2
    }
    
    /// Detect complex rhythmic patterns
    fn detect_complex_rhythm(&self, sequence: &[String]) -> bool {
        let complex_rhythms = sequence.iter()
            .filter(|e| e.as_str() == "SIXTEENTH" || e.contains("SHARP"))
            .count();
        
        complex_rhythms > sequence.len() / 3
    }
    
    /// Detect modal or atonal elements
    fn detect_modal_elements(&self, sequence: &[String]) -> bool {
        // Look for unusual note combinations that suggest modal or atonal music
        let sharp_notes = sequence.iter().filter(|e| e.contains("SHARP")).count();
        let total_notes = sequence.iter().filter(|e| e.len() == 1 || e.contains("SHARP")).count();
        
        total_notes > 0 && sharp_notes as f64 / total_notes as f64 > 0.5
    }
    
    /// Detect highly repetitive patterns
    fn detect_repetitive_patterns(&self, sequence: &[String]) -> bool {
        if sequence.len() < 4 {
            return false;
        }
        
        // Check for exact repetition of patterns
        for pattern_len in 2..=4 {
            for i in 0..sequence.len().saturating_sub(pattern_len * 2) {
                let pattern = &sequence[i..i + pattern_len];
                let next_pattern = &sequence[i + pattern_len..i + pattern_len * 2];
                if pattern == next_pattern {
                    return true;
                }
            }
        }
        false
    }
    
    /// Count compositions with similar musical patterns
    fn count_similar_compositions(&self, target_sequence: &[String]) -> usize {
        self.composition_patterns.values()
            .filter(|sequence| {
                let common_elements = sequence.iter()
                    .filter(|element| target_sequence.contains(element))
                    .count();
                
                common_elements as f64 / sequence.len() as f64 > 0.6
            })
            .count()
    }
    
    /// Generate analysis summary
    pub fn generate_analysis_summary(&self) -> MusicAnalysisSummary {
        let total_compositions = self.analysis_results.len();
        let experimental_compositions = self.analysis_results.iter()
            .filter(|r| matches!(r.complexity_level, ComplexityLevel::Experimental | ComplexityLevel::Complex))
            .count();
        
        let avg_detection_time = self.performance_metrics
            .get("last_detection_time_ms")
            .unwrap_or(&0.0);
        
        let total_elements: usize = self.analysis_results.iter()
            .map(|r| r.musical_sequence.len())
            .sum();
        
        MusicAnalysisSummary {
            total_compositions_analyzed: total_compositions,
            experimental_compositions_found: experimental_compositions,
            average_analysis_time_ms: *avg_detection_time,
            patterns_identified: self.count_unique_patterns(),
            total_elements_analyzed: total_elements,
            style_insights_generated: self.count_style_insights(),
        }
    }
    
    /// Count unique patterns identified
    fn count_unique_patterns(&self) -> usize {
        let mut all_patterns = std::collections::HashSet::new();
        for result in &self.analysis_results {
            for pattern in &result.musical_patterns {
                all_patterns.insert(pattern.clone());
            }
        }
        all_patterns.len()
    }
    
    /// Count style insights generated
    fn count_style_insights(&self) -> usize {
        self.analysis_results.iter()
            .map(|r| r.style_insights.len())
            .sum()
    }
    
    /// Generate common musical patterns for training
    fn generate_common_musical_patterns(&self) -> Vec<Vec<String>> {
        let mut patterns = Vec::new();
        
        // Base common musical patterns
        let base_patterns = vec![
            // Basic chord progressions
            vec![MusicalElement::CMajor, MusicalElement::AMajor, MusicalElement::FMajor, MusicalElement::GMajor],
            
            // Pop song structure
            vec![MusicalElement::Intro, MusicalElement::Verse, MusicalElement::Chorus, MusicalElement::Verse, MusicalElement::Chorus, MusicalElement::Bridge, MusicalElement::Chorus, MusicalElement::Outro],
            
            // Scale patterns
            vec![MusicalElement::C, MusicalElement::D, MusicalElement::E, MusicalElement::F, MusicalElement::G, MusicalElement::A, MusicalElement::B],
            
            // Rhythmic patterns
            vec![MusicalElement::Quarter, MusicalElement::Quarter, MusicalElement::Half, MusicalElement::Quarter, MusicalElement::Rest],
            
            // Minor progression
            vec![MusicalElement::AMinor, MusicalElement::FMajor, MusicalElement::CMajor, MusicalElement::GMajor],
            
            // Jazz progression
            vec![MusicalElement::CMajor, MusicalElement::AMajor, MusicalElement::DMinor, MusicalElement::GMajor],
            
            // Blues pattern
            vec![MusicalElement::C, MusicalElement::E, MusicalElement::G, MusicalElement::B],
            
            // Classical structure
            vec![MusicalElement::Intro, MusicalElement::Verse, MusicalElement::Bridge, MusicalElement::Verse],
        ];
        
        // Extended musical patterns for comprehensive training
        let extended_patterns = vec![
            // Complex chord progressions
            vec![MusicalElement::CMajor, MusicalElement::AMinor, MusicalElement::FMajor, MusicalElement::GMajor, MusicalElement::EMajor, MusicalElement::AMinor, MusicalElement::DMinor, MusicalElement::GMajor],
            
            // Extended song structure
            vec![MusicalElement::Intro, MusicalElement::Verse, MusicalElement::Verse, MusicalElement::Chorus, MusicalElement::Verse, MusicalElement::Chorus, MusicalElement::Bridge, MusicalElement::Solo, MusicalElement::Chorus, MusicalElement::Chorus, MusicalElement::Outro],
            
            // Chromatic scale
            vec![MusicalElement::C, MusicalElement::CSharp, MusicalElement::D, MusicalElement::DSharp, MusicalElement::E, MusicalElement::F, MusicalElement::FSharp, MusicalElement::G, MusicalElement::GSharp, MusicalElement::A, MusicalElement::ASharp, MusicalElement::B],
            
            // Complex rhythmic patterns
            vec![MusicalElement::Eighth, MusicalElement::Sixteenth, MusicalElement::Eighth, MusicalElement::Quarter, MusicalElement::Rest, MusicalElement::Half, MusicalElement::Quarter],
            
            // Modal progression
            vec![MusicalElement::DMinor, MusicalElement::EMinor, MusicalElement::FMajor, MusicalElement::GMajor, MusicalElement::AMinor, MusicalElement::BMinor, MusicalElement::CMajor],
            
            // Jazz ii-V-I
            vec![MusicalElement::DMinor, MusicalElement::GMajor, MusicalElement::CMajor, MusicalElement::CMajor],
            
            // Blues with sevenths
            vec![MusicalElement::C, MusicalElement::E, MusicalElement::G, MusicalElement::ASharp, MusicalElement::C, MusicalElement::F, MusicalElement::A, MusicalElement::C],
            
            // Classical sonata form
            vec![MusicalElement::Intro, MusicalElement::Verse, MusicalElement::Bridge, MusicalElement::Verse, MusicalElement::Bridge, MusicalElement::Solo, MusicalElement::Verse, MusicalElement::Outro],
            
            // Folk song pattern
            vec![MusicalElement::Verse, MusicalElement::Chorus, MusicalElement::Verse, MusicalElement::Chorus, MusicalElement::Bridge, MusicalElement::Chorus],
            
            // Electronic music structure
            vec![MusicalElement::Intro, MusicalElement::Verse, MusicalElement::Chorus, MusicalElement::Solo, MusicalElement::Verse, MusicalElement::Chorus, MusicalElement::Solo, MusicalElement::Outro],
            
            // Ballad structure
            vec![MusicalElement::Intro, MusicalElement::Verse, MusicalElement::Verse, MusicalElement::Chorus, MusicalElement::Verse, MusicalElement::Chorus, MusicalElement::Bridge, MusicalElement::Chorus, MusicalElement::Outro],
            
            // Rock progression
            vec![MusicalElement::EMajor, MusicalElement::AMinor, MusicalElement::CMajor, MusicalElement::GMajor, MusicalElement::FMajor, MusicalElement::CMajor, MusicalElement::GMajor, MusicalElement::EMajor],
        ];
        
        // Genre-specific patterns
        let genre_patterns = vec![
            // Classical baroque
            vec![MusicalElement::CMajor, MusicalElement::GMajor, MusicalElement::AMinor, MusicalElement::EMajor, MusicalElement::FMajor, MusicalElement::CMajor, MusicalElement::GMajor, MusicalElement::CMajor],
            
            // Jazz standard
            vec![MusicalElement::CMajor, MusicalElement::EMajor, MusicalElement::AMinor, MusicalElement::CMajor, MusicalElement::FMajor, MusicalElement::GMajor, MusicalElement::CMajor, MusicalElement::GMajor],
            
            // Blues twelve-bar
            vec![MusicalElement::CMajor, MusicalElement::CMajor, MusicalElement::CMajor, MusicalElement::CMajor, MusicalElement::FMajor, MusicalElement::FMajor, MusicalElement::CMajor, MusicalElement::CMajor, MusicalElement::GMajor, MusicalElement::FMajor, MusicalElement::CMajor, MusicalElement::GMajor],
            
            // Folk melody
            vec![MusicalElement::C, MusicalElement::D, MusicalElement::E, MusicalElement::G, MusicalElement::A, MusicalElement::G, MusicalElement::E, MusicalElement::D, MusicalElement::C],
            
            // Rock anthem
            vec![MusicalElement::EMajor, MusicalElement::BMajor, MusicalElement::AMinor, MusicalElement::EMajor, MusicalElement::CMajor, MusicalElement::GMajor, MusicalElement::DMajor, MusicalElement::EMajor],
            
            // Pop ballad
            vec![MusicalElement::CMajor, MusicalElement::AMinor, MusicalElement::FMajor, MusicalElement::GMajor, MusicalElement::AMinor, MusicalElement::FMajor, MusicalElement::CMajor, MusicalElement::GMajor],
            
            // Country progression
            vec![MusicalElement::GMajor, MusicalElement::CMajor, MusicalElement::GMajor, MusicalElement::DMajor, MusicalElement::GMajor, MusicalElement::CMajor, MusicalElement::DMajor, MusicalElement::GMajor],
            
            // R&B progression
            vec![MusicalElement::AMinor, MusicalElement::FMajor, MusicalElement::CMajor, MusicalElement::GMajor, MusicalElement::AMinor, MusicalElement::FMajor, MusicalElement::GMajor, MusicalElement::CMajor],
        ];
        
        // Rhythmic and structural elements
        let rhythmic_elements = vec![
            vec![MusicalElement::Quarter],
            vec![MusicalElement::Eighth],
            vec![MusicalElement::Half],
            vec![MusicalElement::Whole],
            vec![MusicalElement::Rest],
            vec![MusicalElement::Quarter, MusicalElement::Rest],
            vec![MusicalElement::Eighth, MusicalElement::Eighth],
            vec![MusicalElement::Sixteenth, MusicalElement::Sixteenth, MusicalElement::Eighth],
        ];
        
        // Generate comprehensive training dataset with long sequences
        for iteration in 0..40 {
            // Create long realistic musical composition sequences by combining multiple patterns
            for base_idx in 0..base_patterns.len() {
                let mut long_sequence = Vec::new();
                
                // Start with rhythmic elements for musical foundation
                if !rhythmic_elements.is_empty() {
                    let intro_rhythm = &rhythmic_elements[base_idx % rhythmic_elements.len()];
                    long_sequence.extend(intro_rhythm.iter().cloned());
                }
                
                // Add base pattern
                long_sequence.extend(base_patterns[base_idx].iter().cloned());
                
                // Add 4-6 musical segments to create substantial compositions
                let num_segments = 4 + (iteration % 3);
                for segment in 0..num_segments {
                    // Add rhythmic elements between segments
                    if !rhythmic_elements.is_empty() {
                        let rhythm = &rhythmic_elements[segment % rhythmic_elements.len()];
                        long_sequence.extend(rhythm.iter().cloned());
                    }
                    
                    // Add an extended pattern for complexity
                    let ext_idx = (base_idx + segment) % extended_patterns.len();
                    long_sequence.extend(extended_patterns[ext_idx].iter().cloned());
                    
                    // Add a genre pattern for style
                    let genre_idx = (segment + 1) % genre_patterns.len();
                    long_sequence.extend(genre_patterns[genre_idx].iter().cloned());
                    
                    // Add another base pattern for continuity
                    let next_base_idx = (base_idx + segment + 1) % base_patterns.len();
                    long_sequence.extend(base_patterns[next_base_idx].iter().cloned());
                }
                
                // Add final rhythmic elements for musical closure
                if !rhythmic_elements.is_empty() {
                    let final_rhythm = &rhythmic_elements[iteration % rhythmic_elements.len()];
                    long_sequence.extend(final_rhythm.iter().cloned());
                }
                
                // Convert to strings and add to patterns
                let string_pattern: Vec<String> = long_sequence.iter()
                    .map(|e| e.to_string())
                    .collect();
                patterns.push(string_pattern);
            }
            
            // Create mega-sequences by combining multiple genre patterns
            if iteration % 3 == 0 {
                let mut mega_sequence = Vec::new();
                
                // Combine 5-7 genre patterns into one large musical composition
                let num_patterns = 5 + (iteration % 3);
                for i in 0..num_patterns {
                    let pattern_idx = (iteration + i) % genre_patterns.len();
                    mega_sequence.extend(genre_patterns[pattern_idx].iter().cloned());
                    
                    // Add connecting rhythmic elements and extended patterns
                    if i < num_patterns - 1 {
                        if !rhythmic_elements.is_empty() {
                            let rhythm_idx = i % rhythmic_elements.len();
                            mega_sequence.extend(rhythmic_elements[rhythm_idx].iter().cloned());
                        }
                        let ext_idx = i % extended_patterns.len();
                        mega_sequence.extend(extended_patterns[ext_idx].iter().cloned());
                    }
                }
                
                let string_pattern: Vec<String> = mega_sequence.iter()
                    .map(|e| e.to_string())
                    .collect();
                patterns.push(string_pattern);
            }
        }
        
        // Add realistic full-length composition patterns (long sequences)
        for _composition in 0..25 {
            // Create full-length musical compositions by combining multiple sections
            let mut full_composition = Vec::new();
            
            // Classical symphony movement (20-25 elements)
            full_composition.extend(vec![
                MusicalElement::Intro, MusicalElement::CMajor, MusicalElement::Quarter, MusicalElement::GMajor, MusicalElement::Half,
                MusicalElement::AMinor, MusicalElement::Quarter, MusicalElement::FMajor, MusicalElement::Eighth, MusicalElement::CMajor,
                MusicalElement::Quarter, MusicalElement::GMajor, MusicalElement::Half, MusicalElement::CMajor, MusicalElement::Bridge,
                MusicalElement::DMinor, MusicalElement::Quarter, MusicalElement::GMajor, MusicalElement::Eighth, MusicalElement::CMajor,
                MusicalElement::Half, MusicalElement::AMinor, MusicalElement::Quarter, MusicalElement::FMajor, MusicalElement::CMajor, MusicalElement::Outro
            ]);
            
            // Pop song structure (25-30 elements)
            full_composition.extend(vec![
                MusicalElement::Intro, MusicalElement::Quarter, MusicalElement::Verse, MusicalElement::CMajor, MusicalElement::AMinor,
                MusicalElement::Quarter, MusicalElement::FMajor, MusicalElement::GMajor, MusicalElement::Eighth, MusicalElement::Chorus,
                MusicalElement::CMajor, MusicalElement::Quarter, MusicalElement::FMajor, MusicalElement::GMajor, MusicalElement::Half,
                MusicalElement::AMinor, MusicalElement::Verse, MusicalElement::Quarter, MusicalElement::CMajor, MusicalElement::AMinor,
                MusicalElement::Eighth, MusicalElement::FMajor, MusicalElement::Chorus, MusicalElement::Quarter, MusicalElement::Bridge,
                MusicalElement::DMinor, MusicalElement::Half, MusicalElement::GMajor, MusicalElement::Chorus, MusicalElement::Quarter, MusicalElement::Outro
            ]);
            
            // Jazz standard with improvisation (22-28 elements)
            full_composition.extend(vec![
                MusicalElement::CMajor, MusicalElement::Quarter, MusicalElement::AMinor, MusicalElement::Eighth, MusicalElement::DMinor,
                MusicalElement::Quarter, MusicalElement::GMajor, MusicalElement::Half, MusicalElement::EMajor, MusicalElement::Quarter,
                MusicalElement::AMinor, MusicalElement::Eighth, MusicalElement::FMajor, MusicalElement::GMajor, MusicalElement::Quarter,
                MusicalElement::Solo, MusicalElement::CMajor, MusicalElement::Sixteenth, MusicalElement::AMinor, MusicalElement::Eighth,
                MusicalElement::DMinor, MusicalElement::Quarter, MusicalElement::GMajor, MusicalElement::Half, MusicalElement::CMajor,
                MusicalElement::Quarter, MusicalElement::AMinor, MusicalElement::Eighth, MusicalElement::GMajor, MusicalElement::Rest
            ]);
            
            // Folk ballad with verses and chorus (18-22 elements)
            full_composition.extend(vec![
                MusicalElement::Verse, MusicalElement::C, MusicalElement::Quarter, MusicalElement::D, MusicalElement::Eighth,
                MusicalElement::E, MusicalElement::Quarter, MusicalElement::G, MusicalElement::Half, MusicalElement::Chorus,
                MusicalElement::A, MusicalElement::Quarter, MusicalElement::G, MusicalElement::Eighth, MusicalElement::E,
                MusicalElement::Quarter, MusicalElement::D, MusicalElement::Half, MusicalElement::C, MusicalElement::Verse,
                MusicalElement::Quarter, MusicalElement::C, MusicalElement::D, MusicalElement::Eighth, MusicalElement::E,
                MusicalElement::Rest
            ]);
            
            let string_pattern: Vec<String> = full_composition.iter()
                .map(|e| e.to_string())
                .collect();
            patterns.push(string_pattern);
            
            // Create alternative full-length compositions with different styles
            let mut alt_composition = Vec::new();
            
            // Rock anthem structure (25-32 elements)
            alt_composition.extend(vec![
                MusicalElement::Intro, MusicalElement::EMajor, MusicalElement::Quarter, MusicalElement::BMajor, MusicalElement::Eighth,
                MusicalElement::AMinor, MusicalElement::Quarter, MusicalElement::EMajor, MusicalElement::Half, MusicalElement::Verse,
                MusicalElement::CMajor, MusicalElement::Quarter, MusicalElement::GMajor, MusicalElement::Eighth, MusicalElement::DMajor,
                MusicalElement::Quarter, MusicalElement::EMajor, MusicalElement::Half, MusicalElement::Chorus, MusicalElement::EMajor,
                MusicalElement::Quarter, MusicalElement::BMajor, MusicalElement::Eighth, MusicalElement::AMinor, MusicalElement::Quarter,
                MusicalElement::EMajor, MusicalElement::Solo, MusicalElement::Sixteenth, MusicalElement::EMajor, MusicalElement::Quarter,
                MusicalElement::BMajor, MusicalElement::Eighth, MusicalElement::AMinor, MusicalElement::Quarter, MusicalElement::EMajor,
                MusicalElement::Half, MusicalElement::Outro
            ]);
            
            // Electronic dance music (20-26 elements)
            alt_composition.extend(vec![
                MusicalElement::Intro, MusicalElement::Eighth, MusicalElement::Sixteenth, MusicalElement::Eighth, MusicalElement::Quarter,
                MusicalElement::Verse, MusicalElement::CMajor, MusicalElement::Eighth, MusicalElement::AMinor, MusicalElement::Sixteenth,
                MusicalElement::FMajor, MusicalElement::Eighth, MusicalElement::GMajor, MusicalElement::Quarter, MusicalElement::Chorus,
                MusicalElement::Sixteenth, MusicalElement::Eighth, MusicalElement::Sixteenth, MusicalElement::Quarter, MusicalElement::Solo,
                MusicalElement::Eighth, MusicalElement::Sixteenth, MusicalElement::Eighth, MusicalElement::Quarter, MusicalElement::Chorus,
                MusicalElement::Sixteenth, MusicalElement::Eighth, MusicalElement::Quarter, MusicalElement::Outro, MusicalElement::Rest
            ]);
            
            let alt_string_pattern: Vec<String> = alt_composition.iter()
                .map(|e| e.to_string())
                .collect();
            patterns.push(alt_string_pattern);
        }
        
        patterns
    }
}

/// Music analysis summary data
#[derive(Debug)]
pub struct MusicAnalysisSummary {
    pub total_compositions_analyzed: usize,
    pub experimental_compositions_found: usize,
    pub average_analysis_time_ms: f64,
    pub patterns_identified: usize,
    pub total_elements_analyzed: usize,
    pub style_insights_generated: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 MUSIC PATTERN ANALYSIS");
    println!("==========================");
    println!();
    
    // Initialize analyzer
    let mut analyzer = MusicPatternAnalyzer::new()?;
    
    // Train on common patterns
    analyzer.train_on_common_patterns()?;
    println!();
    
    // Analyze different musical scenarios
    println!("🔍 ANALYZING MUSICAL COMPOSITIONS");
    println!("=================================");
    
    // Scenario 1: Common chord progression
    println!("\n📊 Scenario 1: Common Chord Progression");
    let common_progression = vec![
        MusicalElement::CMajor,
        MusicalElement::AMajor,
        MusicalElement::FMajor,
        MusicalElement::GMajor,
    ];
    
    let result = analyzer.analyze_composition(&common_progression, "song_001", 0.3)?;
    if let Some(analysis) = result {
        println!("✅ Complexity Level: {}", analysis.complexity_level.to_string());
        println!("   Patterns: {:?}", analysis.musical_patterns);
        println!("   Insights: {:?}", analysis.style_insights);
    }
    
    // Scenario 2: Experimental composition
    println!("\n🚨 Scenario 2: Experimental Composition");
    let experimental = vec![
        MusicalElement::CSharp,
        MusicalElement::FSharp,
        MusicalElement::ASharp,
        MusicalElement::DSharp,
        MusicalElement::Sixteenth,
        MusicalElement::Rest,
        MusicalElement::GSharp,
    ];
    
    let result = analyzer.analyze_composition(&experimental, "song_002", 0.3)?;
    if let Some(analysis) = result {
        println!("⚠️  Complexity Level: {}", analysis.complexity_level.to_string());
        println!("   Creativity Score: {:.3}", analysis.creativity_score);
        println!("   Patterns: {:?}", analysis.musical_patterns);
        println!("   Insights: {:?}", analysis.style_insights);
    }
    
    // Scenario 3: Pop song structure
    println!("\n🎵 Scenario 3: Pop Song Structure");
    let pop_structure = vec![
        MusicalElement::Intro,
        MusicalElement::Verse,
        MusicalElement::Chorus,
        MusicalElement::Verse,
        MusicalElement::Chorus,
        MusicalElement::Bridge,
        MusicalElement::Chorus,
        MusicalElement::Outro,
    ];
    
    let result = analyzer.analyze_composition(&pop_structure, "song_003", 0.3)?;
    if let Some(analysis) = result {
        println!("⚠️  Complexity Level: {}", analysis.complexity_level.to_string());
        println!("   Patterns: {:?}", analysis.musical_patterns);
        println!("   Similar Compositions Found: {}", analysis.similar_compositions_found);
        println!("   Insights: {:?}", analysis.style_insights);
    }
    
    // Scenario 4: Complex rhythmic pattern
    println!("\n🥁 Scenario 4: Complex Rhythmic Pattern");
    let complex_rhythm = vec![
        MusicalElement::Sixteenth,
        MusicalElement::Eighth,
        MusicalElement::Sixteenth,
        MusicalElement::Rest,
        MusicalElement::Sixteenth,
        MusicalElement::Sixteenth,
        MusicalElement::Quarter,
    ];
    
    let result = analyzer.analyze_composition(&complex_rhythm, "song_004", 0.3)?;
    if let Some(analysis) = result {
        println!("⚠️  Complexity Level: {}", analysis.complexity_level.to_string());
        println!("   Patterns: {:?}", analysis.musical_patterns);
        println!("   Insights: {:?}", analysis.style_insights);
    }
    
    // Generate summary
    println!("\n📊 MUSIC ANALYSIS SUMMARY");
    println!("=========================");
    let summary = analyzer.generate_analysis_summary();
    println!("Compositions Analyzed: {}", summary.total_compositions_analyzed);
    println!("Experimental Compositions Found: {}", summary.experimental_compositions_found);
    println!("Average Analysis Time: {:.2}ms", summary.average_analysis_time_ms);
    println!("Patterns Identified: {}", summary.patterns_identified);
    println!("Total Elements Analyzed: {}", summary.total_elements_analyzed);
    println!("Style Insights Generated: {}", summary.style_insights_generated);
    
    Ok(())
}
