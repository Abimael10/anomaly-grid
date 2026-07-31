

# Anomaly Grid

     █████╗ ███╗   ██╗ ██████╗ ███╗   ███╗ █████╗ ██╗  ██╗   ██╗
    ██╔══██╗████╗  ██║██╔═══██╗████╗ ████║██╔══██╗██║  ╚██╗ ██╔╝
    ███████║██╔██╗ ██║██║   ██║██╔████╔██║███████║██║   ╚████╔╝ 
    ██╔══██║██║╚██╗██║██║   ██║██║╚██╔╝██║██╔══██║██║    ╚██╔╝  
    ██║  ██║██║ ╚████║╚██████╔╝██║ ╚═╝ ██║██║  ██║███████╗██║   
    ╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝╚═╝   
    [ANOMALY-GRID v0.6.0] - SEQUENCE ANOMALY DETECTION ENGINE

[![Crates.io](https://img.shields.io/crates/v/anomaly-grid.svg)](https://crates.io/crates/anomaly-grid)
[![Downloads](https://img.shields.io/crates/d/anomaly-grid.svg)](https://crates.io/crates/anomaly-grid)
[![Documentation](https://docs.rs/anomaly-grid/badge.svg)](https://docs.rs/anomaly-grid)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Una biblioteca de Rust que implementa cadenas de Markov de orden variable con **interpolación Witten-Bell** para la detección de anomalías en secuencias sobre alfabetos finitos. La fuerza de la anomalía combina la sorpresa por símbolo y el contenido de información, ambos en bits, comprimidos por `tanh` en `[0, 1)`.

## Inicio Rápido

```toml
[dependencies]
anomaly-grid = "0.6"
```

### Detecta un ataque de escalada de privilegios en sesiones de usuario

Entrena con sesiones de usuario benignas, luego escanea sesiones desconocidas en paralelo y muestra solo las ventanas que superan tu tolerancia.

```rust
use anomaly_grid::{AnomalyDetector, batch_score};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let s = |w: &[&str]| -> Vec<String> { w.iter().map(|x| x.to_string()).collect() };

    // 30 benign sessions, three legitimate workflow shapes.
    let mut detector = AnomalyDetector::new(3)?;
    let mut benign = Vec::new();
    for _ in 0..30 {
        benign.extend(s(&["LOGIN", "AUTH", "READ", "WRITE", "READ", "LOGOUT"]));
        benign.extend(s(&["LOGIN", "AUTH", "READ", "READ", "WRITE", "LOGOUT"]));
        benign.extend(s(&["LOGIN", "AUTH", "WRITE", "READ", "READ", "LOGOUT"]));
    }
    detector.train(&benign)?;

    // Score four unknown sessions in parallel (lock-free over &detector).
    let candidates = vec![
        s(&["LOGIN", "AUTH", "READ", "WRITE", "READ", "LOGOUT"]),
        s(&["LOGIN", "AUTH", "READ", "READ", "WRITE", "LOGOUT"]),
        s(&["LOGIN", "AUTH", "WRITE", "READ", "READ", "LOGOUT"]),
        s(&["LOGIN", "AUTH", "PRIV_ESCALATE", "EXFIL", "LOGOUT"]), // attack
    ];
    let results = batch_score(&detector, &candidates, 0.3)?;

    for (i, anomalies) in results.iter().enumerate() {
        if anomalies.is_empty() {
            println!("session {i}: clean");
        } else {
            for a in anomalies {
                println!(
                    "session {i}: ANOMALY in {:?} (strength {:.3})",
                    a.sequence, a.anomaly_strength
                );
            }
        }
    }
    Ok(())
}
```

Salida:

```text
session 0: clean
session 1: clean
session 2: clean
session 3: ANOMALY in ["LOGIN", "AUTH", "PRIV_ESCALATE", "EXFIL"] (strength 0.481)
session 3: ANOMALY in ["AUTH", "PRIV_ESCALATE", "EXFIL", "LOGOUT"] (strength 0.543)
```

Las tres sesiones benignas se limitan a una fuerza de 0.096 en cada ventana, por lo que el umbral `0.3` las despeja. La sesión de escalada de privilegios contiene los símbolos nunca antes vistos `PRIV_ESCALATE` y `EXFIL`, y ambos cuatro-gramas que los tocan superan el umbral.

## Qué Hace Esta Biblioteca

- **Modelo de Markov de orden variable** con retroceso suave Witten-Bell
  `λ(c) = N(c) / (N(c) + T(c))`. El caso base de orden-0 es Laplace
  (Add-α) sobre el alfabeto global: los contextos no vistos nunca colapsan a
  probabilidad cero.
- **Puntuación basada en teoría de la información** en bits en todo el proceso: sorpresa por ventana `(−1/(n−1)) Σ log₂ P` y contenido de información por símbolo `−log₂ P(xᵢ | context)`.
- **Almacenamiento consciente de la memoria**: internador `StateId(u32)` respaldado por
  `Arc<str>`, trie de arena indexado por `NodeId(u32)` con
  hijos `SmallVec<[(StateId, NodeId); 4]>` (≤ 4 en línea).
  El enum `TransitionCounts` permanece en línea como `SmallVec<[(StateId, usize); 4]>`
  para el caso típico y recurre a `HashMap` solo cuando un contexto
  excede 4 continuaciones distintas.
- **Puntuación por lotes en paralelo** (`batch_score`) sobre un
  `&AnomalyDetector` compartido mediante rayon. Sin bloqueos durante la puntuación; determinista
  independientemente del tamaño del grupo de hilos.
- **Linter estrictos**: compila bajo
  `#![deny(clippy::pedantic, clippy::nursery, clippy::unwrap_used,
  clippy::expect_used)]`.
- **Invariantes probados con propiedades**: sumas de probabilidad = 1, entropía
  acotada, determinismo paralelo, ida y vuelta Unicode, finitud de secuencias
  largas.

## Configuración

```rust
let config = AnomalyGridConfig::default()
    .with_max_order(4)?              // Higher order = longer context, more memory
    .with_smoothing_alpha(0.5)?      // Lower = more sensitive to training data
    .with_weights(0.8, 0.2)?         // (likelihood + information) — must sum to 1.0
    .with_memory_limit(Some(100_000))?; // Cap at 100k context nodes (default: 1_000_000)

let detector = AnomalyDetector::with_config(config)?;
```

## Casos de Uso

`anomaly-grid` es adecuado cuando tus datos son una **secuencia de tokens discretos sobre un alfabeto finito** y dispones de un corpus de ejemplos de comportamiento normal conocidos para entrenar. El detector marca las ventanas cuya verosimilitud de Markov bajo ese corpus cae bruscamente: *transiciones* localmente improbables, incluso cuando cada token individual es legítimo.

### Ajustes Concretos

- **Registros de protocolo / máquinas de estados** — estados de sesiones TCP,
  acoplamientos de aplicaciones, rondas de consenso. Detecta sesiones que
  omiten pasos de acoplamiento, golpean transiciones ilegales o se reinician a mitad de transmisión.
  Ver [`examples/network_protocol_analysis.rs`](examples/network_protocol_analysis.rs)
  (flujo tipo TCP de 16 estados).

- **Monitoreo de llamadas al sistema / registros de auditoría** — `open → read → close`,
  `socket → connect → send`. Revela malware sin archivos, escapes a shell,
  y patrones de escalada de privilegios cuyas llamadas individuales son
  legítimas pero cuyo *orden* no lo es. El inicio rápido anterior es una
  versión mínima de esto.

- **Flujos de trabajo operativos** — pasos de libros de ejecución, ordenamiento de pipelines CI,
  macros de sesión CLI. La desviación de la secuencia canónica es la
  señal en sí. Ver [`examples/communication_protocol_analysis.rs`](examples/communication_protocol_analysis.rs)
  (protocolo de comunicaciones de 12 símbolos con ataques inyectados).

- **Exploración de motivos en bioinformática** — tripletes de codones en un marco de lectura conocido, patrones de residuos en un taxón curado. Los desplazamientos de marco
  y las variantes de empalme raras se revelan como ventanas de baja verosimilitud. Ver
  [`examples/protein_folding_sequences.rs`](examples/protein_folding_sequences.rs)
  (alfabeto de 20 residuos).

### Dónde no encaja

- Datos continuos o de alta dimensión (imágenes, audio crudo, vectores de características densos) sin discretización.
- Alfabetos superiores a ~1000 símbolos con `max_order` alto: la memoria del árbol de contextos crece como `|Σ|^max_order`.
- Dependencias de largo alcance más allá de 4–5 tokens. Si la señal reside en
  ventanas de contexto de docenas de tokens, prefiere un modelo de secuencia basado en Transformer (TFT, Anomaly Transformer) o un HMM con estado oculto explícito.

## Pruebas

El conjunto de pruebas está organizado por característica en `tests/`, con fixtures compartidos en `tests/common/mod.rs`:

| Archivo | Cobertura |
|---|---|
| `api.rs` | Pruebas de humo de API pública (constructores, entrenamiento, métricas, optimización) |
| `math.rs` | Invariantes Markov + Kolmogorov + Witten-Bell + entropía de Shannon + KL |
| `detection.rs` | Contrato de detección de anomalías (límites de puntuación, monotonía, umbral) |
| `sequences.rs` | Comportamiento de secuencias (truncamiento de ventanas, escalado de alfabeto, entradas largas) |
| `workflow.rs` | Escenarios de dominio de extremo a extremo (redes, fraude, IoT, syslog) |
| `errors.rs` | Cobertura de rutas de error (variantes de `AnomalyGridError`) |
| `concurrency.rs` | Aserciones estáticas `Send + Sync` + determinismo paralelo |
| `proptest.rs` | Pruebas de propiedades (suma a 1, límites de entropía, Unicode, secuencia larga) |
| `regression.rs` | Regresiones de errores pasados |
| `perf_*.rs` | Rendimiento / memoria / escalabilidad: ejecutar con `--release` |

```bash
cargo test                      # all tests
cargo test --test math          # one suite
cargo test --release perf_      # performance suites
cargo run --release --example network_protocol_analysis
cargo run --release --example communication_protocol_analysis
cargo run --release --example protein_folding_sequences
```

## Documentación

- [docs.rs/anomaly-grid](https://docs.rs/anomaly-grid) — referencia de rustdoc
- [docs/api-reference.md](docs/api-reference.md) — mapa de superficie pública
- [docs/mathematical-implementation.md](docs/mathematical-implementation.md) — Witten-Bell + entropía + KL
- [docs/performance-guide.md](docs/performance-guide.md) — dimensionamiento, poda, puntuación paralela
- [examples/](examples/) — demostraciones ejecutables (redes, comunicaciones, proteínas)
- [CHANGELOG.md](CHANGELOG.md) — historial de versiones

## Licencia

MIT — ver [LICENCE](LICENCE).
