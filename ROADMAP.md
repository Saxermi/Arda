# Arda — Master Development Roadmap 2.0

*Compute. Optimize. Understand.*

**Planungsstand:** 24. November 2026 · **Ausgangsversion:** 0.13.1 · **Ziel:** 1.0 · **Dokumentrevision:** 2.0 (konsolidiert)

Dieses Dokument ist ein Entwicklungsplan, kein Bericht über bereits implementierte Features. Die Sparse-Meilensteine S0–S8 sind bis Arda 0.14.3 implementiert. [Arda 0.14.4](docs/MORIA_0.14.4.md) schließt S9 ab. Siehe [Sparse-Solver und Speicherbudgets in 0.14.3](docs/MORIA_0.14.3.md) für die aktuellen APIs und lokalen Messungen. Zahlen mit der Bezeichnung „Ziel“ oder „Modellrechnung“ sind keine Benchmarkergebnisse. Die bisherigen Ziele bleiben bestehen; Resource-Efficient Computing, maximale praktische Hardwareauslastung, Beyond-RAM Computing, mathematische Introspektion, überprüfbare Numerik und verteiltes HPC sind Querschnittsziele. Diese Revision ersetzt die frühere Planungsdatei, nicht deren Ziele. Die Quellcode-Version ist jetzt 0.14.4.

## 1. Mission und Nicht-Ziele

Arda (Adaptive Runtime for Data and Algorithms) ist eine eigenständige, MIT-lizenzierte und forschungsorientierte wissenschaftliche Programmiersprache mit portabler C11-Runtime. Sie soll numerische Mathematik, hochdimensionale Berechnungen, Simulationen, differenzierbare Programmierung und Machine Learning mit hoher Performance und kontrollierbarem Ressourcenverbrauch ermöglichen. Gewöhnliche Laptops ohne dedizierte GPU sind eine zentrale Zielplattform; Workstations und HPC-Systeme sollen dennoch voll genutzt werden können.

- CPU-first, nicht CPU-only. Ein portabler Referenzkernel bleibt verfügbar; optimierte Kernels werden nach Hardware gewählt.
- Algorithmische Effizienz vor Mikrooptimierung: Struktur, Sparsity, Vorkonditionierung, Datenlokalität und adaptive Auflösung zuerst; anschließend SIMD und Parallelität.
- Minimierung unnötiger Datenbewegungen. RAM, Cache, SSD und später Device-Speicher gemeinsam betrachten.
- Keine implizite lossy Quantisierung, Präzisionssenkung, Densifizierung großer Sparse-Objekte oder Änderung des mathematischen Problems.
- Genauigkeit und Reproduzierbarkeit sind getrennte Konfigurationen von Geschwindigkeit. Exakte mathematische Umformungen dürfen in Floating Point Rundungsunterschiede erzeugen; bitweise Reproduzierbarkeit ist eine strengere Option.
- Optionales Offloading ist kein Ersatz für effiziente In-RAM-Kernels. SSD darf niemals der Standard sein, wenn RAM und Laufzeit einen besseren Pfad erlauben.
- Kein Anspruch auf allgemein schnellere Ausführung als NumPy, Julia, PETSc oder llama.cpp. Wettbewerbsvorteile müssen durch faire Messungen auf realen Workloads belegt werden.
- Kein Versprechen, beliebige Supercomputerprobleme oder 70B-Pretraining auf einem Laptop praktikabel zu machen. Große Inference und Fine-Tuning sind vom vollständigen Training zu unterscheiden.

### 1.1 Positionierung und Zielgruppen

**Arda is an independent, research-driven scientific programming language for numerical mathematics, resource-aware execution, and mathematical transparency. Its development targets hardware from everyday laptops to distributed research clusters.**

Zielgruppen: Studierende, Lehrende, Ingenieure, Forschende, wissenschaftliche ML-Entwickler und Entwickler eingebetteter numerischer Anwendungen. Der erste externe Nutzen soll bereits mit Sparse-Numerik, mathematischer Dokumentation und CPU-Optimierung sichtbar werden, nicht erst in 1.0. Python/SciPy, Julia/SciML, MATLAB/Mathematica, C/C++/Fortran und PETSc/llama.cpp sind Referenzen, nicht pauschal zu ersetzende Systeme. Differenzierung: einheitliche mathematische Objekte, Ressourcenplanung, nachvollziehbare Algorithmen und überprüfbare Resultate. Benchmark für wissenschaftliche Produktivität neben Rohperformance: Zeit und Codeumfang vom Modell bis zum reproduzierbaren Bericht.

### 1.2 Verbindliche Querschnittsziele

1. Maximale praktische CPU-Performance bei gleicher mathematischer Aufgabe und Genauigkeit.
2. Skalierbare numerische Mathematik, Sparse, strukturierte und komprimierte Operatoren.
3. Beyond-RAM mit kontrolliertem I/O; SSD als zusätzliche Speicherebene, nicht als kostenlose Rechenleistung.
4. Differenzierbare Programmierung, wissenschaftliches ML und ressourcenbewusste Inference/Training.
5. Wissenschaftliche Simulation und adaptive Verfahren.
6. Explainable & Reproducible Scientific Computing: getTex, Kosten, Annahmen, Verifikation, Experimente.
7. HPC: Multicore, NUMA, optionale GPU, MPI und verteilte Operatoren.
8. Kompakte, portable, zugängliche und einbettbare Runtime; Tolkien-Identität mit englischen Alternativen.

## 2. Nachweisbarer Ausgangsstand und offene technische Schulden

Basis: Archiv `arda-0.13.1-maintenance.zip`, `MAINTENANCE.md`, `ROADMAP_0.14.md`, `src/arda.h`, `src/tensor.c`, `src/scientific.c`, `Makefile`.

- Vorhanden: Lexer, Parser, Bytecode-Compiler, VM, inkrementeller Mark-and-Sweep-GC, Float64-Tensoren mit Rang bis 8, Views, skalare Autodiff, SIMD-Matmul, FFT, spezialisierte matrixfreie Poisson-Operatoren, RK45, L-BFGS.
- Aktueller `Obj.as.array`: `double *data`, `int rank, rows, cols`, `size_t n`, `int shape[8]`, `size_t stride[8]`, `Obj *owner`. `O_ARRAY` und `O_AD` sind vorhanden, native Sparse-Objekte nicht.
- Dichte Arrays sind auf 16.000.000 Elemente begrenzt. 256³ = 16.777.216 ist bereits darüber. Die Grenze wird nicht blind entfernt; später durch geprüfte logische Größen, Adressierungs- und Materialisierungsgrenzen und Budgets ersetzt.
- `VM.bytes` ist keine vollständige Peak-RSS- oder Bytecode-Speichermessung. Frontend-OOM ist noch fatal; keine Sicherheits-Sandbox. Bytecode-GC arbeitet gruppenweise.
- Dokumentierte Messung 0.13.1: jeweils 126 reguläre und 8 Referenztests in Standard-, BLAS/LAPACK- und Sanitizer-Build; Linux Debian 13/GCC 14.2. Ubuntu und ARM64 wurden dort nicht separat validiert. Diese Ergebnisse werden vor 0.14 erneut reproduziert.
- Bestehender Makefile-Glob `src/*.c`, separate Build-Verzeichnisse für BLAS/LAPACK/Sanitizer, inkrementelle Objektdateien, eingebettete Module. Verzeichnisumbauten müssen Build- und Formatregeln mitändern.
- Querschnitt ab sofort: lokale Testautomatisierung (keine GitHub Actions), dokumentierte Sprach- und API-Verträge, Regressionstests, Fehlerpfade, Sanitizer, Fuzzing und unabhängige Numerik-Referenzen. Kein Aufschieben bis 0.28.

## 3. Architekturverträge

### 3.1 Ebenen und Zuständigkeiten

Core: Lexer, Parser, Compiler, VM, GC und öffentliche Werte. Numerik: mathematische Repräsentationen und Algorithmen. Arda Memory Engine (AME): Speicherbudget, Buffer-Lebensdauer, Pooling, Dateispeicher. Arda Kernel Engine (AKE): Dispatch, native Kernels, SIMD, Threads, externe Backends. Adaptive Compute Engine (ACE): Planung, Kostenmodelle, Graphen, Ausführungsentscheidungen. Arda Machine Learning (AML): Autodiff, Modelle, Training, Inference. Arda Mathematical Documentation Engine (AMDE, Codename Rúmil): mathematische Metadaten, MathIR, Renderer, Verträge und optionale Traces. HPC-Schicht: Topologie, Kommunikation, Partitionierung und verteilte Operatoren; benutzt AME/AKE/ACE statt paralleler Doppelarchitekturen. Diese Bereiche wachsen inkrementell, keine sofortige Neuimplementierung.

### 3.2 Orthogonale Eigenschaften

Shape (logische Größe), dtype (skalare Semantik), representation (dense, CSR, COO, low-rank, TT, operator), encoding (plain, Q8/Q4 block encoding), layout (strided, tiled, contiguous), storage (heap, mapped, chunked), device (CPU, iGPU, GPU) und evaluation (eager, graph, matrix-free) sind getrennt. Ein Q4-Tensor kann gleichzeitig disk-backed sein. Ein Low-Rank-Operator besteht aus mehreren Tensoren und ist nicht einfach ein Bytebuffer. Nicht jede Kombination ist zulässig; unterstützte Kombinationen werden per Capability-Registry geprüft.

### 3.3 Eigentum, Fehler und Threading

- GC-Objekte besitzen oder referenzieren explizite Buffer-Handles. Views halten den Besitzer am Leben. Shared Buffer werden genau einmal im Budget erfasst. Spätere Zero-Copy-Exporte pinnen den Besitzer bis zur Freigabe.
- `size_t`-Multiplikationen/Additionen, Indexkonvertierungen, Alignment und Dateioffsets vor Allokation prüfen. Logische Größe ist nicht gleich physisch allokierte Größe.
- Kein `longjmp` aus Worker-Threads. Kernels liefern Status, die VM meldet den Fehler nach Synchronisation. Speicherreservierungen, Pins, temporäre Dateien und Locks haben definierte Cleanup-Pfade.
- VM und GC bleiben zunächst single-threaded. Worker arbeiten ausschließlich an gepinnten nativen Buffern, erzeugen keine VM-Objekte. Keine GC- oder VM-Reentranz ohne expliziten Vertrag.
- Fehler bei Budgetüberschreitung müssen möglichst abfangbar sein; echtes OS-OOM kann davon abweichen. Kein Versprechen eines absoluten Prozesslimits allein durch interne Zähler.

### 3.4 Ressourcenpolitik

Zähler für logisch repräsentierte Bytes, aktive native Allokationen, reservierte Pool-Kapazität, temporären Workspace, mapped range, disk footprint, aktuelle und maximale Budget-Nutzung getrennt halten. Extern: Peak RSS, cgroup memory.peak, page faults, I/O und Swap messen. Für Linux-Tests harte Grenzen optional mit cgroup v2 und ausgeschaltetem Swap; berücksichtigt auch relevante Dateicaches. Allocation accounting allein ist kein Prozess-RAM-Messwert.

Vorgeschlagene API (erst nach entsprechender Implementierung): `compute.configure({"memory_limit": "2 GiB", "disk_limit": "80 GiB", "threads": 4, "objective": "speed", "allow_offload": nae, "allow_approximation": law})`. Genauigkeit/Toleranz und Determinismus separat einstellen. `compute.explain` zeigt Plan und Schätzungen, `compute.stats` tatsächliche Zähler. Budget vor Ausführung inklusive Output und Workspace reservieren, bei Fehler freigeben. Nicht unterstützte Operationen kontrolliert ablehnen; keine stillschweigende Densifizierung.

### 3.5 Zero-Cost Scientific Introspection

Unbenutzte Zusatzfunktionen dürfen den heißen numerischen Pfad nicht mit Pro-Element-Prüfungen, Trace-Allokationen, AD-Knoten oder Formelerzeugung belasten. Dispatch einmal pro Operation; gewöhnliche Tensoren bleiben gewöhnliche Buffer. Native mathematische Deskriptoren liegen separat; kompakte benutzerdefinierte MathIR-Metadaten optional beim Compile-Schritt. Metadaten-Modi none/compact/full (API-Entwurf); none darf nicht vorgeben, einen verlorenen AST rekonstruieren zu können. Auch Startup, Compile-Zeit, Binary-Größe und RSS messen; „zero cost“ ist eine Zielsetzung, keine absolute Garantie. Sicherheits-, Typ-, Index- und Overflow-Prüfungen bleiben aktiv. Keine globale fast-math-Freigabe durch Speed-Modus.

Vier Pfade: normal/fast (keine optionale Aufzeichnung), static analysis (keine numerische Ausführung), verified (explizite Ergebnisprüfung), traced (gezielt instrumentiert). Profile Fast/Adaptive/Verified/Research sind Komfortkonfigurationen; Genauigkeit, Determinismus, Budget und erlaubte Approximation bleiben separat. Fast muss nicht auf effiziente, gecachte Standardpläne verzichten. Ein vollständiger didaktischer Trace darf einen anderen Kernel verwenden, muss dies aber explizit kennzeichnen.

### 3.6 Mathematical Introspection und Provenance (AMDE / Rúmil)

Drei getrennte Informationsquellen: (A) mathematische Semantik der Funktion, (B) dokumentierter Algorithmus eines konkreten Backends, (C) tatsächlich beobachtete Ausführung. MathIR beschreibt Mathematik, nicht zwingend Gleitkomma-Reihenfolge. Eine symbolische Identität ist kein Nachweis einer tatsächlich erfolgten Auswertung. Dynamische Calls, Mutation, unbekannte Typen und externe Bibliotheken können die Abdeckung begrenzen. Unbekanntes explizit kennzeichnen statt Formeln zu erfinden. `cos` aus libm darf beispielsweise nicht als Taylorreihen-Implementierung bezeichnet werden, nur weil die Reihe eine korrekte Identität ist.

| Detailstufe | Inhalt | Numerische Ausführung nötig? |
|---|---|---|
| 0 | Kompakte mathematische Definition | Nein |
| 1 | Quellnahe mathematische Zwischenoperationen | Nein |
| 2 | Shape, Konstanten, Einheiten, Voraussetzungen, Abhängigkeiten | Nein, soweit statisch bekannt |
| 3 | Interne mathematische Algorithmen, Rekursionen, unterstützte Herleitungen | Nein; konkreter adaptiver Pfad ggf. erst nach Planung bekannt |
| 4 | Konkrete Werte, ausgewählte Zwischenschritte, Konvergenz | Ja; nur aus explizit aufgezeichnetem Trace oder vorliegenden Resultaten |

`f.getTex(0..3)` führt f niemals aus. Level 4 ist auf einem Run/Trace-Objekt verfügbar; ohne Run meldet es fehlende Daten, löst nicht heimlich eine Rechnung aus. `getTex` liefert LaTeX-String; `getMath` strukturierten Bericht, `getMathML` Webnotation. Renderer escapen Identifier, Strings und Unicode; Dokumentexport führt keine untrusted LaTeX-Shell-Befehle aus. MathIR für Konstante, Variable, Arithmetik, Calls, Tensorindizes, Summen, Piecewise, unterstützte Schleifen/Recurrences; Tiefe, Knotenzahl und Output-Größe begrenzen. Native Deskriptoren pro Algorithmus/Backend mit Version, Voraussetzungen, Komplexität, Formel, Abdeckungsstatus. Annotations/Registry für Fremdmodule. Compiler-AST wird heute freigegeben; MathIR muss vorher selektiv extrahiert oder aus explizit verfügbaren Quellen erzeugt werden. GC besitzt Metadaten mit Proto-Lebensdauer; keine unbegrenzten REPL-Roots.

| API-Entwurf | Bedeutung und Grenzen |
|---|---|
| f.getTex(level), f.getMath(), f.getMathML(level) | Statische mathematische Dokumentation; keine Ausführung |
| f.getAlgorithm(), f.getAssumptions(), f.getComplexity() | Bekannte Verfahren und Voraussetzungen, Worst/typical falls begründet; bei Dispatch mehrere Kandidaten |
| f.getCost(spec), f.getPlan(spec) | Shape/DType-basierte FLOPs, RAM/SSD, I/O, Kostenintervalle, Algorithmusplan; unbekannte Datenabhängigkeit melden |
| f.getUnits(spec), f.getDerivative() | Statische Dimensionsanalyse bzw. unterstützte symbolische Ableitung |
| f.getSensitivity(args) | Numerische lokale Sensitivität, JVP/VJP; normierte Sensitivität nur bei sinnvoller Referenz |
| f.verify(args, result) | Optionale zusätzliche Residuen-/Fehlerprüfung; keine universelle mathematische Garantie |
| f.trace(args, options) | Einmalige instrumentierte Ausführung, Run mit Ergebnis und Ereignissen |
| run.getTex(4), run.getAlgorithm(), run.getPlan() | Tatsächlicher Pfad, gemessene vs. geschätzte Werte, Trace-Abdeckung |
| compute.record / reproduce / compare / sweep | Experiment-Manifest, Wiederholung, fairer Vergleich, budgetierte Parameterstudien |
| getScaling / getCommunication / getTopology / getPartition / getEfficiency | Plan- oder Run-spezifische HPC-Daten mit expliziter Unterscheidung geschätzt/gemessen |

Trace-Vertrag: Aus, Summary, Sampled, Detailed (nur kleine Probleme). Grenzen für Events, Tiefe, Bytes, Laufzeit, Tensor-Samples, Ausgabe und Disk; Sampling pro Iteration und bevorzugt bereits berechnete Residuen. Bei Limit standardmäßig Trace kürzen und Rechnung weiterführen, Bericht markiert Auslassungen; optional abbrechen. Kein vollständiger Tensor-Snapshot ohne Zustimmung, keine globale Speicherung aller VM-Instruktionen, kein unkontrolliertes Wiederholen von Seiteneffekten. Bei 1 Mio. Unbekannten, 1.000 Iterationen: vier Float64-Vektoren je Iteration = 32 GB Rohdaten, acht Skalare je Iteration = 64 kB (ohne Metadaten). Maximalbudget z. B. 16 MiB. HPC: lokale Summary plus reduzierte globale Kennzahlen, begrenzter Trace je Rank, kein Gather großer Zustände an Rank 0.

### 3.7 Numerische Verlässlichkeit und wissenschaftliche Typen

- Normierte Residuen, Rückwärtsfehler, Konditionsschätzungen, Rekonstruktionsfehler und Gradchecks; kleines Residuum allein beweist keine kleine Vorwärtsabweichung. Solver-Ergebnis trägt Konvergenzstatus, Ursache und Toleranz. Vertragsprüfungen `require`/`ensure` sind explizit; `assume` ist eine dokumentierte Anwenderzusicherung, kein Beweis. Unsichere Voraussetzungen können zu einem abgelehnten Plan führen. SPD z. B. ist nicht bloß Symmetrie. Validierungen nie stillschweigend auslassen, wenn sie vertraglich gefordert sind.
- `units`: optionale Dimensionsobjekte (SI, Umrechnung, Winkel, affine Temperaturen separat), Einheit einmal pro homogenem Tensor; keine Kosten für normale Tensoren. Mischdimensionierte Vektoren nur mit expliziter Struktur; Addition inkompatibler Dimensionen ablehnen. `getUnits` symbolisch soweit bekannt.
- `uncertainty`: lineare Fehlerfortpflanzung mit AD, Kovarianz JΣJᵀ, unabhängige/korrelierte Eingaben, explizite Annahmen; keine dichte Kovarianz bei übergroßen Problemen erzwingen. Monte Carlo, streaming und reproduzierbare Seeds später. Messunsicherheit nicht mit Rundungsfehler verwechseln.
- `exact` (rationale/big integer) und `interval` (gerichtete Rundung, validierte Einschließungen; optionale MPFR/GMP-Backends) als eigene Typen. Intervallgrenzen und Domänen prüfen, kein fast-math für garantierte Einschließung. Adaptive Precision zunächst für validierte iterative refinement; explizite Fehlergröße und Fallback.
- Experiment-Manifest: Code-/Eingabe-Hashes, Version, Seed/Generatorzustand, DType, Toleranz, Backend, Plan, Threads/Ranks, Hardware, externe Bibliotheken, Prüfergebnisse. Hashing großer Dateien opt-in und budgetiert, keine unbeabsichtigte Veröffentlichung von Daten. Reproduktion erfordert Inputs und kompatible Versionen; bitweise/numerische/statistische Modi separat.
- Persistenter Checkpoint-Restart für unterstützte Solver/Simulation/Training: Zustand, Schritt, RNG, Eingabe-/Versionsprüfung, atomare Datei, Prüfsumme, Disk-Quota; kein universelles Einfrieren der VM versprechen. Nicht mit AD-Recompute-Checkpointing verwechseln.

### 3.8 HPC-Verträge von Anfang an

CPU-first, nicht CPU-only: serieller Pfad und identische mathematische API für unterstützte Operationen. 32-Bit lokale Indizes wo möglich, globale Dimensionen/Offsets/Indizes später 64 Bit; lokale Sparse-Formate nicht unnötig verbreitern. Globale Formate und Verteilung getrennt von lokalen Puffern. Operationen auf verteilten Objekten sind kollektiv, nicht implizit netzwerkfähig für beliebige VM-Objekte. MPI-Kommunikatoren, Besitz, Rank-Fehler und kollektive Aufrufreihenfolge explizit; keine MPI-Aufrufe aus GC-Finalizern. Nichtblockierende Requests pinnen Buffer bis completion; MPI-Threadlevel prüfen, standardmäßig FUNNELED (nur initialisierender Thread ruft MPI). Fehler über Ranks koordinieren, kein einzelner Rank verlässt eine kollektive Operation still. MPI-Fehler nicht als allgemein recoverable versprechen. Partitionen, Halo/ghost, lokale und globale Reduktionen, deterministische Reihenfolge optional. Ressourcenbudgets pro Rank und pro Node, externe BLAS-Threads koordinieren; NUMA first-touch, optionale Affinity und hwloc. ACE darf nur zugewiesene Rechenknoten verwenden; keine automatische Clusterbereitstellung. Netzwerk-Latenz, Kommunikationsvolumen, Synchronisation und Lastungleichgewicht im Kostenmodell.

## 4. Meilensteine und Unterversionen

Technische Titel benennen das Ziel jeder Version. Unterversionen übernehmen den Entwicklungscodenamen ihrer Versionsreihe, sofern kein eigener Codename angegeben ist.

### Arda 0.14 — Resource-Efficient Sparse Computing

*Development codename: Moria*

*Planungspriorität: P0, sofort.*

#### Arda 0.14.0 — Sparse Matrix Foundations

*Development codename: Moria*

**Status** — Implementiert.

Baseline reproduzieren. `O_SPARSE` oder zwei explizite Objekttypen, `src/sparse.c/.h`, eigener nativer ID-Bereich ohne Kollision mit bisherigen Modulen. GC-Freigabe, Typanzeige, Formatierung, Modulregistrierung, Native-Dispatch. Intern geprüfte Größenarithmetik und Sparse-Allokation mit Reservierung/Fehlercleanup. Float64-Werte, uint32-Indizes, size_t-Zeilenoffsets; auf 64-Bit-Systemen 8 Byte pro Offset. Nullbasierte Indizes, leere/rechteckige Matrizen, endliche Werte. `sp.coo`, `sp.csr`, `sp.crs`, `sp.shape`, `sp.nnz`, `sp.bytes`, `sp.format`, `sp.get`, `sp.components`. Komponenten als Kopien oder unveränderliche Views, nie unkontrolliert veränderbare CSR-Struktur.

**Implementierte Konvertierung** — COO → CSR ohne Dense: Zeilen zählen, Prefix-Sum, Einträge verteilen, pro Zeile stabil nach Spalte sortieren, Duplikate in dokumentierter Reihenfolge addieren, exakte Nullen entfernen, Offsets kompakt neu schreiben. Alternativ globales Sortieren als Referenz. Ziel O(rows + nnz + sum(k_i log k_i)), sortierter Input schneller; O(rows + nnz) Workspace, keine Kopie in Dense. Endliche Summen prüfen. CSR → COO linear. CSR-Konstruktor verlangt streng steigende Spalten je Zeile, gültige Offsets und keine strukturell gespeicherten Nullen. COO erlaubt explizite Nullen und Duplikate; `nnz` zählt gespeicherte Einträge. `sp.from_dense`, `sp.todense` mit Vorabprüfung des gesamten Outputs.

#### Arda 0.14.1 — Sparse Matrix Assembly

*Development codename: Moria*

**Status** — Implementiert.

`sp.builder`, `sp.append`, `sp.extend` und `sp.finish` verwenden typisierte native Puffer. Batches werden vor der Einfügung geprüft. Kapazitätswachstum berücksichtigt die gleichzeitig belegte alte und neue Kapazität. Erfolgreiche Finalisierung gibt kanonisches CSR zurück und schließt den Builder. `sp.eye`, `sp.diags` und `sp.tridiag` erzeugen CSR direkt.

#### Arda 0.14.2 — Sparse Algebra and Linear Operators

*Development codename: Moria*

**Status** — Implementiert.

`sp.matvec` und `sp.spmm` verwenden einen seriellen CSR-Kernel mit bei Bedarf vorgeschalteter COO-Kanonisierung. `sp.transpose`, `sp.add`, `sp.scale`, `sp.diagonal`, `sp.read_mm` und `sp.write_mm` unterstützen Sparse-Algebra und Matrix Market (coordinate real general/symmetric). Die interne Schnittstelle `LinearOperator` bindet Dense, CSR und den matrixfreien Poisson-Operator an. `sp.estimate` meldet Form, gespeicherte Einträge und den zusätzlichen Matvec-Workspace.

#### Arda 0.14.3 — Sparse Solvers and Memory Budgets

*Development codename: Moria*

**Status** — Implementiert.

`sp.cg` und `sp.pcg` lösen SPD-Systeme mit optional wiederverwendbarem Workspace und berechnen das wahre Schlussresiduum. `sp.verify` prüft Lösungen separat. `sp.set_budget` und `sp.memory` begrenzen und erfassen verwaltete Sparse- und Solver-Puffer. Ergebnisse und Grenzen sind in der [0.14.3-Spezifikation für Sparse-Solver und Speicherbudgets](docs/MORIA_0.14.3.md) dokumentiert. Der direkte tridiagonale Solver gehört zu [0.15.0](#arda-015--numerical-methods-and-mathematical-documentation).

**Querschnitt:** Versionierte statische mathematische Deskriptoren decken Konvertierung, Sparse-Algebra und Solver ab. Mathematische Verträge und Komplexitätsangaben werden mit der abschließenden 0.14.4-Validierung konsolidiert. Globale und lokale Indizes bleiben getrennte Konzepte.

#### Arda 0.14.4 — Sparse Computing Release Validation

*Development codename: Moria*

**Status** — Lokal validiert. [Messwerte und Reproduktion](docs/MORIA_0.14.4.md)

Matrix Market, Dokumentation, ASan/UBSan/LSan, GC-Stress, automatisierte SciPy-Randomtests, Peak-RSS und Laufzeit für Builder, Conversion, SpMV und CG getrennt. Bestehende 126+8 Baseline-Tests weiterhin grün. Ziel: 1 Mio. Zeilen tridiagonal, 2.999.998 gespeicherte Werte, direkter CSR-Aufbau plus SpMV <256 MiB Peak RSS im definierten Linux-Test; COO-Build/Conversion separat messen und dokumentieren. SPD-Test A=tridiag(-1,4,-1) mit erzeugtem b=A*x_ref, relatives Residuum ≤1e-8 und normierter Lösungsfehler dokumentiert, Ziel <512 MiB. Keine Dense-Materialisierung; kein Anheben des alten Dense-Limits nur für Sparse.

**Beispiel:** A=[[0,5,0,-1],[0,0,0,0],[4,0,0,7]]. Unsortierte COO: rows=[2,0,0,2,0], cols=[3,1,1,0,3], values=[7,2,3,4,-1]. CSR: offsets=[0,2,2,4], cols=[1,3,0,3], values=[5,-1,4,7]. Rückkonvertierung: rows=[0,0,2,2]. Matvec mit [1,2,3,4] ergibt [6,0,32]. Ein zusätzliches Paar (1,2,4) und (1,2,-4) muss beim Kanonisieren verschwinden. Für 1 Mio. Zeilen: Dense 8 TB, COO 47.999.968 B, CSR 43.999.984 B (etwa 41,96 MiB), ohne Header und Workspace.

### Arda 0.15 — Numerical Methods and Mathematical Documentation

*Development codename: Rúmil*

- **0.15.0:** LU mit partieller Pivotisierung, determinant, trace, diag, inv als Solve, Cholesky, Householder-QR, tridiagonaler Thomas-Algorithmus nur unter geeigneten Stabilitätsbedingungen; banded solve; bestehende `solve`-API erhalten. Rekonstruktion und Residuen für ill-conditioned/indefinite Fälle, Singulärität melden.
  - **Status** — Implementiert und lokal validiert. Rekonstruktion, normierte Residuen und Fehlermeldungen sind im [Release-Bericht](docs/RUMIL_0.15.0.md) dokumentiert. Der bisherige Vektoraufruf von `numc.solve` ist weiterhin verfügbar. [Numerische Verträge](research/implemented/linear-algebra.md#reusable-float64-factorizations)
- **0.15.1:** SVD, Pseudoinverse, Rank, Konditionsschätzung, `eigh`, `eig` mit optional LAPACK und dokumentierter Verfügbarkeit; portable Fallbacks nur wenn validiert. Komplexe Eigenwerte als komplexe Skalare bzw. Real/Imag-Ausgabe bis 0.16.
  - **Status** — Implementiert und lokal validiert. `eigh` und `eig` ergänzen die SVD-Funktionen im optionalen LAPACK-Build. `eig` liefert Eigenpaare als Real- und Imaginärteile. [Release-Bericht](docs/RUMIL_0.15.1.md)
- **0.15.2:** PCG- oder anderer spezifizierter PRNG, explizite Generatorobjekte, Seeds, Uniform/Normal/Integer/Shuffle/Choice/Poisson, Statistik (Median, Quantile, Kovarianz, Korrelation, Histogramm, Regression, cumsum, argmin/max, clip); Online-Welford und stabile Summen. Zufallsfolge versionsgebunden, keine Krypto-Eignung versprechen.
  - **Status** — Implementiert und lokal validiert. `random` bietet explizite PCG32-Generatoren, `stats` bietet deskriptive Statistik und Online-Momente. [Release-Bericht](docs/RUMIL_0.15.2.md)
- **0.15.3:** Komplexe Skalare, `conj`, Betrag, Argument, komplexe Elementfunktionen. Low-rank Matrixoperator U,V; Randomized SVD mit Oversampling, Power-Iteration und Fehlerprüfung; kein vollständiger Dense-Output für Operator-Eingaben. Kronecker/diagonal/band/Toeplitz-Operatoren. Zero-order Optimierung (finite Differenzen, zufällige Richtungen, optional MeZO-artige Experimente) im opt-Modul.
  - **Status** — Implementiert und lokal validiert. `complex` und `linop` laufen im portablen Build. `linalg.rsvd` verwendet optional LAPACK für die reduzierte Zerlegung. `opt.zeroth_order` bietet Koordinatendifferenzen und reproduzierbare Zufallsrichtungen. [Release-Bericht](docs/RUMIL_0.15.3.md)
- **Arda 0.15.4 — Mathematical Documentation (AMDE v1)** — Descriptor-Registry, LaTeX-Renderer, `getTex(0..3)`, `getAlgorithm`, `getAssumptions`, `getComplexity` für native Mathematik; einfache arithmetische benutzerdefinierte Funktionen vor AST-Freigabe extrahieren. `getCost` für bekannte Dimensionen und Kernels ohne Ausführung. Tests für Operatorpräzedenz, Escape, gebundene/unbekannte Calls, `cos`/libm-Grenze, Abschaltung und GC. Symbolische Gleichung und Algorithmus nicht verwechseln. Beispiel `caro f(x): nan x*x+2` → f(x)=x²+2. `nc.solve.getTex(3)` beschreibt Pivot, Eliminationsfaktoren und Rückwärtseinsetzen, nicht explizite Inverse als Implementierung. Wiederverwendbare LU/QR/Cholesky-Faktorobjekte, mehrere rechte Seiten; separate Lebensdauer und Mutation.
  - **Status** — Implementiert und lokal validiert. `getCost` schätzt Arithmetik und Puffer für dichte Lösungen, Matrix- und Skalarprodukte sowie CSR-Matrix-Vektor-Produkte. Geradlinige Nutzerfunktionen erhalten begrenzte MathIR-Metadaten mit `--mathdoc=compact` (Standard). Level 0 vereinfacht ausgewählte skalare Ausdrücke für die Darstellung. Level 1 und 3 zeigen die Rechenschritte des Quelltexts. [Funktionsverträge](research/implemented/mathdoc.md) · [Release-Bericht](docs/RUMIL_0.15.4.md)
- **0.15.5 — Mathematical library:** Gamma/Beta/Bessel/erf, stabile Spezialfunktionen, Nullstellensuche (Brent/Newton), Interpolation/Splines, adaptive Quadratur, erweiterte Verteilungen und Statistik. Numerische Referenzen inkl. Randdomänen; kein kompletter CAS. Bei 1.000 rechten Seiten einer 1.000²-Matrix ist LU-Reuse theoretisch etwa 251× weniger FLOPs als 1.000 Faktorisierungen, kein gemessener Speedup.
  - **AMDE v1 — begrenzte statische Schleifenanalyse:** einfache `for`-Schleifen über erkennbare `range`-Grenzen mit skalarem Akkumulator als Summen, Produkte oder einfache explizite Rekurrenzen in MathIR und `getTex(0..3)` darstellen. Startwert, Schrittweite, exklusive/inklusive Grenzen, leere Iterationsbereiche und Abhängigkeiten korrekt abbilden; mathematische Darstellung von der tatsächlichen Floating-Point-Auswertungsreihenfolge unterscheiden. Keine Schleifenausführung, kein Unrolling nach Laufzeitwerten und keine dynamischen Seiteneffekte bei statischen Abfragen. Nicht sicher erkennbare Grenzen, Mutation/Aliasing, dynamische Calls, `break`/`continue` und nicht unterstützte Schleifen konservativ als unbekannt markieren (`mun` gemäß Query-Vertrag). Tests für Summen, Produkte, Rekurrenzen, Randfälle, Nichtausführung, Metadatenlimits und GC. Allgemeine `while`-Schleifen, komplexer Kontrollfluss, rekursive Analyse und tatsächliche Ausführungstraces bleiben in 0.22.4.
  - **Status** — Implementiert und lokal validiert. Die skalare Bibliothek enthält reelle Spezialfunktionen, abgesicherte Nullstellensuche, lineare und natürliche kubische Interpolation, adaptive Gauss–Kronrod-Quadratur und erweiterte Verteilungen und Statistik. AMDE erkennt eine einzelne halboffene native `range`-Schleife mit einem skalaren Akkumulator und konstanter von null verschiedener Schrittweite. [Release-Bericht](docs/RUMIL_0.15.5.md)
- **Gate:** QR-Rekonstruktion und Orthogonalität, Cholesky für SPD und Fehlermeldung für ungeeignete Matrizen, SVD-Spektren gegen LAPACK, Low-Rank-Approximation mit gemessenem Fehler. Beispiel 100.000×100.000 rank-50: 80 GB dense, 80 MB für U,V in Float64; O(nr) Matvec. Keine Behauptung, jede Matrix habe niedrigen Rang.
- **Abschlussaufgabe — Arda-Wiki.** Nach Abschluss von 0.15.5 einschließlich der statischen Schleifenanalyse und vor Beginn von 0.16 ein englischsprachiges GitHub-Wiki mit Schnellstart, Sprach- und Modulübersicht sowie AMDE-Beispielen für native Funktionen, Nutzerfunktionen und einfache Schleifen veröffentlichen. Für API-Details und ausführbare Beispiele auf die Dokumentation im Repository verweisen und das Wiki in der [README](README.md) verlinken.
  - **Status** — Veröffentlicht. Die Beispiele wurden lokal mit Arda 0.15.5 ausgeführt. [Arda-Wiki](https://github.com/Saxermi/Arda/wiki)

### Arda 0.16 — Virtual Tensors and Memory Management (AME v1)

*Development codename: Mithril*

- **0.16.0:** Buffer-Handle mit Größe, Alignment, Ownership und Release-Funktion; bestehende O_ARRAY-API über Adapter erhalten. Shape, dtype, strides, offset, logical size, device, encoding separat. View-Aliasing, readonly, negative/zero strides, kontrollierte Mutation, sichere Pinning-Regeln. Keine globalen ABI-Zusagen vor 1.0.
  - **Status** — Gemeinsame CPU-Buffer mit eigenen Release-Funktionen, kompatible Float64-Adapter, schreibgeschützte Aliase sowie geprüfte negative und Null-Strides verfügbar. Native Zugriffe bleiben synchron. [Puffer- und View-Verträge](research/implemented/typed-tensors.md#buffers-and-views)
- **0.16.1:** DType registry: bool, uint8, int32/int64, float32/64, complex64/128; FP16/BF16 zunächst Storage/Conversion mit portablem Fallback. Explizite Typ-Promotion, Akkumulationsdtype, Integeroverflow und Cast-Policy. Komplexe Tensoren/FFT, batched matmul, axis reductions, mask/gather/scatter, einsum/tensordot in Etappen. Quantisierung als eigene Encoding-Schicht vorbereiten.
  - **Status** — Die DType-Registry, portable Arithmetik, batched matmul, Achsensummen, mask/gather/scatter, komplexe FFT und erste einsum/tensordot-Formen sind implementiert. Blockquantisierung ist in 0.18.3 geplant. [DType-Verträge](research/implemented/typed-tensors.md#dtypes-and-arithmetic)
- **0.16.2:** `mmap` read-only, überprüfte Header/Shape/Endian/Dateigröße/Offsets; write/create nur mit klarer Ownership und Flush-Vertrag. Chunked File-Storage mit explizitem Blockzugriff. Keine Annahme, mmap kontrolliere die RSS exakt. Externes Trunkieren von Dateien als dokumentiertes Risiko (SIGBUS), robuste Streaming-Reader für unzuverlässige Dateien. Disk-Budget und atomare temporäre Dateien, sichere Cleanup-Strategie.
  - **Status** — Schreibgeschützte Abbildung, explizite Blockleser, Headerprüfung und atomare Dateierstellung mit Byte-Budget verfügbar. [Dateiformat und Validierung](research/implemented/typed-tensors.md#tensor-files)
- **0.16.3:** Buffer Pool, Workspace Reservation, Speicherstatistik, Blockiteratoren, read/write tile, `compute.memory_stats`. Erster Budget-Plan für bekannte Operatoren; übergroße Outputs wahlweise explizit disk-backed oder abgelehnt. Logische Größen und physische Limits trennen, rank>8 erst nach sicherer Metadatenverwaltung. Nicht alle Operationen auf mmap/chunked gleichzeitig unterstützen.
  - **Status** — Implementiert. Pufferbudgets und wiederverwendbarer Workspace, Blockiteratoren, Tile-Zugriff und ein Budgetplan für Streaming-Mittelwert und -Varianz sind verfügbar. Zu große RAM-Ausgaben werden vor der Allokation abgelehnt. Die Messungen mit 20 GB und 80 GB bleiben offen. [Speicher- und Streaming-Verträge](research/implemented/typed-tensors.md#memory-budgets-and-streaming)
- **0.16.4:** Low-rank Tensorformate: CP, Tucker, Tensor Train (TT), zunächst Konstruktion aus Kernen/Faktoren, Elementzugriff, norm/dot, einfache Kontraktion, kontrolliertes Rounding und Explizitmaterialisierung. TT-SVD für passende Eingaben, TT-cross/adaptive approximative Verfahren später. Komprimierte Repräsentation nicht durch vorheriges Anlegen eines unpassend großen Dense-Tensors erzwingen.
  - **Status** — CP, Tucker und Tensor Train unterstützen Zugriff und direkte Kontraktionen. TT-SVD und TT-Rounding verwenden optional LAPACK. [Numerische Verträge](research/implemented/low-rank-tensors.md)
- **0.16.5 — Scientific types:** `units` v1, Dimensionsprüfung einmal pro Tensoroperation, SI/Umrechnungen/Temperaturfälle. Optional rationale/big integer und Intervallprototypen separat; garantierte Intervalle erst nach Rundungsvalidierung stabil. MathIR Tensor-/Indexnotation und DType/Shape-Verträge; globale 64-Bit-Dimensionen, lokale kompakte Indizes. Plain Tensor-ABI und Kernel ohne Units-Overhead erhalten.
  - **Status** — SI-Größen, Dimensionsprüfung, Temperaturumrechnung, Shape-/DType-Verträge und MathIR-Indexnotation sind implementiert. Logische Spezifikationen nutzen 64-Bit-Dimensionen. Konkrete Tensorlayouts behalten kompakte Indizes. Rationale Zahlen, Big Integer und garantierte Intervalle bleiben separate Vorhaben. [Verträge für wissenschaftliche Typen](research/implemented/scientific-types.md)
- **Gate:** 0.13.1/0.14-Kompatibilität, View-Lebensdauer, Readonly und Aliasing, Randfälle, typed references, 20-GB Streaming-Reduktion Ziel <512 MiB Peak RSS, 80-GB Dateitensor ausgewählte Operationen Ziel <1 GiB. SSD-Datei und physische I/O getrennt dokumentieren; keine Sparse-Datei als alleinigen Durchsatznachweis verwenden.

### Arda 0.17 — Memory-Efficient Differentiable Programming

*Development codename: Vilya*

- **0.17.0:** Reverse-Mode-Graph für Tensoren; op tape, explizite Gradientenakkumulation, topologische Reihenfolge, gespeicherte Tensorversionen; Mutation/Aliasing erkennen. Scalar-AD kompatibel. Gradienten für arithmetic, sin/cos/exp/log, sum/mean, matmul, reshape/transpose, broadcasting und slicing.
  - **Status** — Der Entwicklungsstand unterstützt Reverse-Mode-Autodiff für dichte Float64-Tensoren, prüft Änderungen über gemeinsame Puffer und übernimmt Gradienten nach einem erfolgreichen Durchlauf. Lazy Range, strukturierte Fehler und benannte Solver-Ergebnisse sind verfügbar. [Vilya-Verträge](docs/VILYA_RUNTIME.md)
- **0.17.1:** `backward`, `grad`, `zero_grad`, `detach`, `no_grad`, VJP, gezielte JVP/Forward Mode und Jacobian/Hessian in Unterstufen. Sparse-Matvec-Gradienten zunächst für Werte bei festem Sparsity-Pattern (Indizes nicht differenzierbar). Solver-Adjoints später mit transponiertem Operator und Konvergenzprüfung.
  - **Status** — `zero_grad`, `detach`, `no_grad`, VJP, JVP, dichte Jacobi- und Hesse-Matrizen und Ableitungen für CSR-Matvec mit festen Indizes sind implementiert. Die Abfragen nutzen den ausgeführten Tape. Solver-Adjoints bleiben eine spätere Erweiterung. [Verträge](docs/VILYA_RUNTIME.md#gradient-queries-and-sparse-matvec)
- **0.17.2:** Liveness-Analyse, sofortige Freigabe nicht mehr benötigter Tape-Daten, wiederverwendbare Gradient-Buffer, Gradient Accumulation mit korrekter Batch-Gewichtung, checkpoint/recompute für reine oder reproduzierbare Operationen. RNG-Zustände bei Checkpointing sichern; kein erneutes Ausführen externer Seiteneffekte.
  - **Status** — Optionaler Backward-Durchlauf mit Freigabe und Wiederverwendung temporärer Gradienten, Freigabe des Tape am gewählten Output und stichprobengewichteter Akkumulation ist implementiert. Explizite Checkpoints spielen dichte elementweise Operationen erneut ab und behalten aufgezeichnete Zufallswerte. [Verträge und lokale Messung](docs/VILYA_RUNTIME.md#memory-efficient-backward-and-checkpoints)
- **0.17.3:** Aktivierungs-Offloading in Chunked/SSD-Storage, budgetierter Speicherplan, optional selektive Checkpoints; Profiling von zusätzlich erforderlichen Flops und I/O. In-place/fused optimizer update erst nach korrekter Behandlung von Shared Parameters, Grad-Clipping und mehreren Gradientenbeiträgen.
  - **Status** — Explizites Offloading dichter Float64-Aktivierungen mit Dateibudget, begrenzten Lesefenstern und optionalen Checkpoint-Grenzen ist implementiert. Erfasst werden Dateibytes, Lesevorgänge und skalare Replay-Operationen. In-place/fused Updates bleiben an die genannten Voraussetzungen gebunden. [Verträge und lokale Messung](docs/VILYA_RUNTIME.md#activation-offloading)
- **0.17.4 — Sensitivity & uncertainty:** `getSensitivity(args)`, relative Sensitivität soweit definiert, JVP/VJP-basierte Unsicherheitsrechnung, strukturierte Kovarianz, Monte-Carlo-Prototyp; HVP statt volle Hessian bei großen n. MathIR Kettenregel und unterstützte symbolische Ableitung; numerischer AD-Graph und Dokumentationsgraph getrennt. Beispiel m=10±0,2 kg, a=9,81±0,05 m/s²: F=98,1±2,0247 N bei unabhängigen Standardunsicherheiten und linearer Näherung. 1 Mio. Parameter: volle Float64-Hessian 8 TB, HVP ohne Materialisierung. Verteilte Gradienten-/Parameterbesitz-Verträge vorbereiten, noch kein MPI.
  - **Status** — Numerische Sensitivitätsberichte für skalare Ergebnisse, Hesse-Vektor-Produkte, lineare Unsicherheitsfortpflanzung mit Diagonal- und Faktorkovarianz, deren Vektorprodukte und ein reproduzierbarer Gauß-Monte-Carlo-Prototyp sind implementiert. MathIR erzeugt statische Kettenregel-Ableitungen für unterstützte skalare Ausdrücke. Ein Entwurf legt Parameteridentität, samplegewichtete Gradientenreduktion und Abnahmetests für spätere MPI-Arbeit fest. [Verträge und Validierung](docs/VILYA_RUNTIME.md#sensitivity-and-uncertainty) · [Entwurf für verteilten Parameterbesitz](research/planned/computing-systems.md#draft-distributed-parameter-and-gradient-ownership-contract)
- **Gate:** Analytische, finite-difference- und unabhängige PyTorch/JAX-Referenzen; Garbage-Collection/Graph-Lifetime; gleichwertige Gradienten bei Checkpointing innerhalb Toleranz. Beispiel 1.000×8 MB Zustände: vollständige Speicherung 8 GB; 50er-Blöcke: 21 Checkpoints inkl. Start ≈168 MB plus bis ca. 400 MB Segmentzustände und übriger Workspace (nicht bloß 168 MB Peak).
  - **Status** — A1 lokal mit 1.000 Schritten und 8 MB pro Zustand abgeschlossen. Der Peak RSS sinkt von 7,48 GiB auf 543 MiB und die maximale absolute Gradientendifferenz beträgt `1,44e-15`. Numerische Referenzen und GC-Tests sind dokumentiert. [A1-Benchmark](research/implemented/optimization-and-ad.md#vilya-a1-checkpoint-benchmark)

### Arda 0.18 — High-Performance CPU Execution and Adaptive Planning (AKE v1, ACE v1)

*Development codename: Shadowfax II*

- **0.18.0:** Reproduzierbare Microbenchmarks (GEMM, GEMV, SpMV, AXPY, reduction, sin/cos, strided copies, VM loops); `perf stat`, Cache- und Bandbreitenmessung, Laufzeit-CPU-Erkennung. Portable Kernel-Referenz. GCC/Clang LTO/PGO als optionale Builds; portable Artefakte nicht pauschal mit `-march=native` verteilen.
  - **Baseline für dichtes Float64-`matmul`:** Standardbuild ohne BLAS und optionalen OpenBLAS-Build mit Eigen und NumPy vergleichen. Quadratische Matrizen (128 × 128, 512 × 512, 1024 × 1024) mit identischen Eingaben und zunächst einem Thread messen. Hardware, Compilerflags und Backend-Versionen dokumentieren. Median wiederholter Messungen nach Warmup angeben, ohne Eingabeerzeugung und Programmstart, inklusive Ausgabeallokation. Ergebnisse gegen eine unabhängige Referenz mit dokumentierten numerischen Toleranzen prüfen.
  - **Status** — Reproduzierbare Ein-Thread-Baseline für Arda (portabel und OpenBLAS), Eigen und NumPy gemessen. Die übrigen CPU-Mikrobenchmarks, CPU-Erkennung und optionale Build-Rezepte sind verfügbar. [Messungen und Reproduktion](research/implemented/transforms-and-kernels.md#0180-cpu-baseline)
- **0.18.1:** Kernel registry mit x86 SSE2/AVX2/FMA und ARM64 NEON; optional AVX-512/VNNI/AMX/SVE nur auf unterstützten Geräten und nach Benchmark. Aligned/unaligned Zugriffe, Tail Handling, Cache Blocking, vector math für sin/cos/exp. Akkumulationspräzision und FMA-Kontraktion unter numerischen Verträgen. BLAS als optionaler Pfad, externer Threadcount koordinieren.
  - **Abnahmekriterium für dichtes Float64-`matmul`:** Der Standardbuild ohne BLAS muss bei den 512 × 512- und 1024 × 1024-Fällen gegenüber der 0.18.0-Baseline unter gleichen Messbedingungen reproduzierbar schneller sein. Die Verbesserung muss über der Messstreuung liegen. Numerische Toleranzen einhalten und das Regressionsziel von höchstens 10 % für unveränderte kritische Workloads anwenden, einschließlich des kleinen 128 × 128-Falls.
  - **Status** — Laufzeitwahl für SSE2, AVX2 und AArch64 NEON, unveränderte Reihenfolge im Standardpfad und explizites FMA. Optionales libmvec für gewöhnliche Float64-Operationen. Die lokale portable GEMM-Baseline wurde bei allen drei Größen geschlagen. [Messungen](research/implemented/transforms-and-kernels.md#0181-register-blocked-matrix-multiplication)
- **0.18.2:** Native Thread-Pool oder zunächst OpenMP, konfigurierbare Threadzahl, Schwellen für Parallelität, statische Verteilung für gleichmäßige Kernels und Work Stealing erst für heterogene Aufgaben. Kein Threadstart pro kleinem Kernel, keine BLAS-Oversubscription, VM/GC single-threaded. Laptop-Energie/Thermik-Profile messen, Affinity nur optional.
  - **Status** — Optionales OpenMP für große dichte Float64-Matrixprodukte und AXPY mit statischer Aufteilung und einem Thread als Standard. CBLAS nutzt die Threadsteuerung des Providers. Lokale Zwei-Thread-Validierung ist dokumentiert. Energie- und Thermikmessungen auf einem Laptop stehen aus. [Messungen und Grenzen](research/implemented/transforms-and-kernels.md#0182-controlled-cpu-parallelism)
- **0.18.3:** Q8/Q4 block encoding, Blockskalen, Metadaten, quantize/dequantize, quantisierte GEMV/GEMM ohne Voll-Dekompression; portable Referenz und SIMD. FP16/BF16 compute bei Hardwareunterstützung. Quantisierungsfehler und End-to-End-Qualität separat prüfen; Q2/ternary experimentell. Beispiel Q4: 32 Werte + 4 B scale =20 B, 5 bit/Parameter effektiv; 70B entsprechend 43,75 GB, nicht 35 GB.
  - **Status** — Explizite Q4/Q8-Blöcke mit 32 Werten und Float32-Skalen, GEMV/GEMM ohne vollständige Dekompression, portable und SIMD-Pfade sowie Float16/BFloat16 mit Float32-Akkumulation sind implementiert. Speicher, Laufzeit und Vorhersagequalität sind getrennt gemessen. [Verträge und Messungen](research/implemented/quantization.md)
- **0.18.4:** Prefetch und Double Buffering (POSIX pread/Threads, optional madvise; io_uring erst bei messbarem Nutzen). Tiled GEMM/GEMV, Streaming-Reduktion, einfache explizite Fusion (`axpy`, `mul_add`, `map_reduce`) und erste optimierte Graphregionen. Puffer-Pool beachten. Strikter Modus verbietet unzulässige Reassociation.
  - **Status** — Explizite Float64-Kernel für geteilte GEMM/GEMV und fusionierte Ausdrücke, Streaming-Reduktion und optionales Doppelpuffer-Prefetch. [Verträge](research/implemented/cpu-compute.md#0184-bounded-kernels)
- **0.18.5:** ACE v1 für explizit unterstützte Operationen: Kernelwahl, Blockgröße, Threadzahl, Workspace-Schätzung; kleine plan/explain-Schnittstelle. Kostenmodell `max(F/P, B_ram/bw_ram, B_ssd/bw_ssd)` als optimistische Untergrenze, plus Overheads und ggf. sequentielle Phasen; reale Messung kalibriert. Autotuning begrenzt, cachebar, überschreibbar, nicht bei jedem Aufruf.
  - **Status** — Float64-GEMM/GEMV-Pläne mit Kernelwahl, Budgetprüfung, gemessener optionaler Suche und parametrisiertem Kostenmodell. [ACE v1](research/implemented/cpu-compute.md#0185-ace-v1-plans)
- **0.18.6 — NUMA & overhead controls:** optional hwloc, CPU/Cache/NUMA-Topologie, first-touch, Pinning und Threadbudgets; Benchmarks für kleine und große Operationen sowie Oversubscription. `compute.compare` mit gleichen Toleranzen, Warmup, Wiederholungen und getrennten Backend-Parametern. Testen, dass AMDE/Trace/Units/AD deaktiviert keine Pro-Element-Kosten verursachen. Optionale LTO/PGO und Build-Feature-Schalter, keine kombinatorische ungetestete Build-Matrix. ACE v1 zeigt Kernel, Blockgröße, Threads und geschätzten Workspace; tatsächliche Auswahl im Run-Bericht.
- **Gate:** Kein unbewiesener allgemeiner Speedup. Auf festgelegter Hardware und identischen Daten: Ziel ≤10% Regression gegenüber vorigem Release bei unveränderten kritischen Workloads (außer begründeter Sicherheit/Korrektheit); mindestens drei klar dokumentierte Kernelverbesserungen; Zeit, RSS, Datenbewegung, Genauigkeit und Threadscaling angeben. Vergleich mit optimiertem BLAS/SciPy/Julia. 10 Mio. Float64 `a*b+c*d+e`: bis zu 240 MB temporäre Allokationen entfallen, ohne Inputs/Output zu ändern.

### Arda 0.19 — Advanced Numerical Solvers

*Development codename: Aulë*

- **0.19.0:** Einheitlicher Operator- und Solververtrag für dense/sparse/structured/matrix-free; `apply`, `apply_transpose`, dimensions, dtype, workspace. Standardisierte SolverResult inklusive normiertem Residuum, iterations, converged, breakdown, reason. Optional Callback-Operator mit synchroner VM-Ausführung und klaren Kosten.
- **0.19.1:** CG/PCG, MINRES, restarted GMRES (restart begrenzt), BiCGSTAB, LSQR/LSMR, Krylov-workspace-Budgets. Jacobi/Block-Jacobi, ILU(0), incomplete Cholesky nur unter passenden Voraussetzungen. Sparse LU optional (Fill-in- und Speicherprüfung). Eigenwerte mit Lanczos/Arnoldi, Reorthogonalisierung wo nötig.
- **0.19.2:** Geometrisches Multigrid für ausgewählte elliptische Gitteroperatoren: smoother, restriction, prolongation, coarse solve, V-cycle; anisotrope/schwierige Fälle testen. AMG optional über PETSc/andere Bibliotheken später. Matrixfreie Stencils und spezialisierte Randbedingungen.
- **0.19.3:** Mixed-precision iterative refinement mit höherpräzisem Residuum und Fallback, low-rank/Kronecker-Operatoren; experimentelle TT-Solver, Hierarchical Matrices und Sparse Grids separat, falls nachweisbarer Anwendungsnutzen. Kein automatischer approximativer Ersatz ohne Nutzererlaubnis.
- **0.19.4 — Verified & communication-aware solvers:** `verify` für Linear Solve/Eigen/SVD, Fehlermaße und Konditionsschätzung; Solver-Trace mit Iteration, Residuum, Breakdown, Vorkonditionierung, toleranzbezogenen Stops. Adaptive Precision nur wenn explizit erlaubt, prüft geeignetes Fehlerkriterium. Vorbereitung von globaler Reduktion und Halo-Apply; pipelined/communication-avoiding Krylov erst nach Stabilitätsvergleich. `require`/`assume` zunächst als API-Metadaten für Operatoren.
- **Gate:** 128³ Poisson (2.097.152 Punkte), unabhängige Manufactured Solution, relatives Residuum ≤1e-8, Fehler gegenüber exakter diskreter Lösung; Ziel <512 MiB Peak RSS, Laufzeit und Iterationszahl gegen bisheriges CG und externe Referenz. Mehrere Testgrößen zur Skalierung, keine ausschließliche Eigenvektor-Testrechte Seite. Beispiel sechs Float64-Vektoren belegen 96 MiB, weitere Puffer extra. Nichtkonvergenz sauber melden.

### Arda 0.20 — Scientific Simulation

*Development codename: Ulmo*

- **0.20.0:** ODE: vorhandenes RK4/RK45 erhalten, adaptive Toleranzen, dense output, events, Verlet/Leapfrog, saveat/streaming Zustände und Checkpoints; Fehlerkontrolle und Restart.
- **0.20.1:** Implicit Euler, BDF/Radau in Etappen, Newton-Verfahren, Jacobian/JVP, steife Testprobleme; lineare Operatoren und Vorkonditionierung wiederverwenden. Optionale etablierte Backend-Solver statt unvalidierter eigener Implementierungen.
- **0.20.2:** PDE: Poisson, Wärmeleitung, Wellengleichung, Advektion-Diffusion; Dirichlet/Neumann/periodisch, Finite-Difference/Finite-Volume ausgewählte Schemata, Stabilitätsbedingungen, matrixfreie Stencils, Cache-freundliche Domain-Tiles.
- **0.20.3:** FEM: 1D-Linien-/2D-Dreieckselemente, Quadratur, Materialmodelle in einfachem Umfang, COO-Assembly → CSR, Randbedingungen, Sparse-Solver; 3D erst nach Validierung. Adaptive Mesh Refinement mit Fehlerindikatoren, konservativer Übertragung und Ressourcenbudgets. Experimentelle einfache Strömungssolver (keine pauschale CFD-Produktionsreife).
- **0.20.4:** Differenzierbare Simulation: Adjoint/JVP für ausgewählte ODE/PDE-Modelle, Parameteridentifikation, Checkpointing und Streaming der Zeitreihe. Vergleich mit finite differences und analytischen Referenzen.
- **0.20.5 — Experiments & domains:** `compute.sweep` für unabhängige Parameterstudien mit begrenztem Task-Scheduler, Streaming-Ausgabe, Fehlertabelle und sicherer Wiederverwendung unveränderter Faktorobjekte. Persistenter Checkpoint/Resume für ausgewählte ODE/PDE-Solver; dokumentierte Randbedingungen und Diskretisierung in AMDE, mathematische Verträge und Erhaltungssätze. Partitionierbare Gitter, Ghost-Zellen und Subdomain-Verträge als MPI-Vorbereitung. Signalverarbeitung (Faltung, Filter, Wavelets), Geometrie/Rotationen, Regelungstechnik und Sparse-Graphalgorithmen als optionale Erweiterungspakete, keine Voraussetzung für alle Simulationen.
- **Gate:** analytische Lösungen, Gitter-/Zeitschrittverfeinerung, erwartete Konvergenzordnung, Erhaltungssätze und dokumentierte Randbedingungen. Reproduzierbares Wärmeleitungs- und FEM-Beispiel auf 8-GB-Laptop.

### Arda 0.21 — Beyond-RAM Data Processing and Visualization (AME v2)

*Development codename: Khazad-dûm*

- **0.21.0:** Streaming CSV/TSV, JSON, NPY/NPZ, Matrix Market (aus 0.14), VTK, optional HDF5. Sichere Header-/Dateiprüfung, bounded decompression, Endianness und untrusted size validation. Binärformate ohne Vollmaterialisierung, Formatversionen dokumentieren.
- **0.21.1:** Disk-backed Outputs, explizites read/write tile, Chunk-Cache mit Budget, LRU oder passende Zugriffspolitik, Dirty-Block-Writeback, Crash-Cleanup und Quota. Deterministische Blockreihenfolge optional, atomare Checkpoints. Kein allgemeines transparentes SSD-Swapping beliebiger VM-Objekte.
- **0.21.2:** Out-of-core Statistik, Filter, Map/Reduce, Tiled GEMM, ausgewählte QR/SVD (TSQR/randomized SVD) und disk-backed Sparse-SpMV. External sort für sehr große COO-Assembly experimentell; Ein-/Ausgabe-Pässe und Spill-Kosten messen. Disk-backed Ergebnis selbst kann größer sein als RAM.
- **0.21.3:** SVG-Plotting (line, scatter, hist, heatmap), optional PNG, Contour/Vector Field und VTK-Export; große Daten mit explizitem Downsampling oder Streaming für Darstellung, nicht stillschweigend für Berechnung.
- **0.21.4 — Reports & persistent state:** mathdoc/rumil-Export in LaTeX, Markdown, HTML, JSON mit Formeln, Einheiten, Algorithmus, Prüfbericht und optionalem Trace; Ausgabe budgetieren/streamen, nicht große Tensoren dumpen. Versionierte Experimentmanifeste und Checkpoint-Dateien mit Prüfsummen, Quota, atomarem Commit. Vorbereiten partitionierter Dateiformate, MPI-IO erst nach MPI-Grundmodul. Optional TSQR und blockweises External Sort nach Benchmarks.
- **Gate:** echte Datenfiles > RAM (20 GB/80 GB), Cold/Warm getrennt, harte Linux-Testlimits ohne Swap, Ausgabevalidierung und dokumentierter I/O; Ziel Streaming-Statistik <512 MiB, 80-GB-Reduktion <1 GiB. Keine universelle Out-of-Core-Semantik versprechen.

### Arda 0.22 — Language Runtime and Adaptive Execution (ACE v2)

*Development codename: Palantír*

- **0.22.0:** Tupel, Sets, Generatoren/Lazy Range, Iteratoren, Keyword-Argumente, Typannotationen, F-Strings und Context Manager in separat getesteten Paketen. Sprachspezifikation aktualisieren; neue Syntax nicht erzwingen, existierende Programme behalten.
- **0.22.1:** VM-Optimierungen: lokale Slots, Call/Attribute-Caches, spezialisierte Opcodes, GC-Sweep-Latenz, Bytecode-Lebensdauer und Speicherdiagnostik. Profiling vor Änderung. Ressourcenlimit für Instruktionen/Stack/Memory, kontrollierte Host-Funktionen, Abbruch; keine Behauptung einer Sicherheits-Sandbox ohne Isolation und Audit.
- **0.22.2:** Opt-in Lazy Graphs für reine numerische Regionen. Tensorversionen und Aliasing, Materialisierungspunkte und Seiteneffekte definieren; Eager bleibt Standard. Graph-Liveness und Memory Plan, topologische Scheduler, tile-aware Fusion.
- **0.22.3:** ACE v2: Candidate plans (in-RAM, tiled, mapped, structured, optional approximate), Kostenmodell CPU/RAM/SSD, Peak-Workspace-Schätzung und Planwahl unter Budget. Plan erklären, Schätzfehler erfassen, Budgetreserve vor Ausführung, Laufzeit-Fallback nur wenn semantisch erlaubt und I/O/Output konsistent. Planner Overhead bei kleinen Operationen begrenzen.
- **0.22.4 — Contracts & AMDE v2:** Funktionsanalyse für sichere Bedingungen, erweiterte Schleifen/Recurrences (aufbauend auf der begrenzten statischen Schleifenanalyse aus 0.15.5), rekursive Calls mit Tiefenlimit; `require`/`ensure` und `assume` (Syntax optional nach Sprachreview). Symbolische und numerische Semantik getrennt. Vollständiges `getCost/getPlan(spec)` für unterstützte Graphen, Kostenintervalle, unbekannte Pfade, gecachte Pläne; keine numerische Ausführung durch statische Analyse. CPU/RAM/SSD/Device/Netzwerk-Kostenprofile als vorbereitete Capability-Registry. Ausführungsprofile, begrenzter Trace und getAlgorithm auf Run-Objekten. Statische Verträge dürfen dynamische Seiteneffekte nicht ausführen.
- **Gate:** deterministischer Modus, Eager/Lazy-Äquivalenz, absichtliche Mutation und Fehlerreihenfolge, unerreichbare Graphen/Buffer freigeben. Beispiel 100-GB-Datei, sum(sin(x)*exp(-x)): blockweise einlesen, fused compute + reduction, keine vollständigen Zwischentensoren; Fließkomma-Toleranz dokumentieren.

### Arda 0.23 — Scientific Interoperability and MPI Foundations

*Development codename: Eärendil*

- **0.23.0:** `libarda.a`, `libarda.so`, öffentliches opaque C-Handle-API, Version/ABI-Verhandlung, Eval/Call/Errors, Rooting, Pin/Unpin, Buffer export/import, zwei unabhängige VM-Instanzen, dokumentierte Threadregeln. Interne `Obj`-Struktur nicht veröffentlichen. Fatalen OOM/longjmp an Host-Grenzen weiter behandeln.
- **0.23.1:** optional FFI (libffi), explizite Signaturen, Ownership und Fehlervertrag; untrusted FFI nicht erlauben. Host-Callback-Regeln. Python-Adapter mit NumPy-/SciPy-Zero-Copy wo Layout, DType, Endian und Lifetime passen; ansonsten explizite Kopie. DLPack/Buffer Protocol optional.
- **0.23.2:** GGUF/SafeTensors laden (nur unterstützte Formate/Encodings), validierte Offsets/Shape/Metadaten; kein Ausführen von Pickle. Optionale GGML/PETSc/optimierte BLAS-Backends; portable Runtime weiterhin unabhängig. Modellformate nicht an eine bestimmte Inference-Bibliothek binden.
- **0.23.3 — MPI foundations:** optional `MPI=1` gegen Open MPI oder MPICH; MPI_Init_thread/FUNNELED, rank/size, communicators, send/recv, isend/irecv/wait, broadcast, scatter/gather, reduce/allreduce, barrier, split. GC-sichere Request-Lifetime, keine implizite Serialisierung beliebiger Werte, checked counts und große Nachrichten ggf. stückeln. `mpirun -np 4 ./arda job.arda` auf einem Host als erster lokaler Test; Mehrknoten-Validierung später. PETSc-Adapter (Vec/Mat/KSP) als optionaler Proof-of-Concept, nicht Voraussetzung für native lokale Numerik.
- **0.23.4 — Metadata interoperability:** MathML, Descriptor-Registrierung für FFI/Plugins, externe Backend-Provenance; unbekannte libm/BLAS/PETSc-Innenschritte als nicht verfügbar deklarieren. Mathematische Metadaten und öffentliche C-API versionieren. Optionale FFTW, SLEPc, hypre, HDF5, ScaLAPACK nach Bedarf und Lizenzreview, nicht alle gleichzeitig.
- **Gate:** C-Host mit mehreren VMs, Zero-Copy-Ownership-/Mutationstests, Parser-Fuzzing, Fehlermeldungen bei nicht unterstützten Encodings, Kompatibilitätsmatrix und Lizenzprüfung optionaler Backends.

### Arda 0.24 — Optimizing Compiler

*Development codename: Fëanor*

- **0.24.0:** Typisierte numerische IR, Shape/DType-Inferenz, Graph Verification, SSA oder geeignete DAG-IR, Effect-/Alias-Analyse, Lowering zu bestehenden Kernels. Keine komplette Neuschreibung des Bytecode-Compilers.
- **0.24.1:** Constant folding, dead code elimination, CSE, Liveness, Loop Fusion, Tiling, layout-aware contraction planning. Fast-math nur explizit; Floating-Point-Äquivalenz je Modus testen. Maximaler Workspace Teil des Optimierungsziels.
- **0.24.2:** Experimenteller JIT/AOT für eingeschränkte typisierte numerische Funktionen (LLVM oder libgccjit optional), Guard/Fallback zum Interpreter, Code-Cache mit Speichergrenze, sichere Lebensdauer, W^X soweit Plattform unterstützt. Compile-Zeit und Warmup separat messen; kleine Funktionen nicht unnötig kompilieren.
- **0.24.3 — Symbolic & parallel lowering:** begrenzte symbolische Vereinfachung und Ableitung (Domänen, Vorzeichen, Nichtdifferenzierbarkeit, Rewrite-Budget beachten); gemeinsame Semantik mit numerischer IR, nicht beide IRs verschmelzen. `sym` als kleiner algebraischer Kern, kein vollständiges Mathematica. Partitionierbare reine numerische Regionen, Thread- und Kommunikationsbarrieren explizit. Traced vs optimierte Ausführung eindeutig labeln.
- **Gate:** Interpreter/IR/JIT/AOT-Referenztests, Fehlerpfade und deterministischer Modus; Nutzen gegen optimierte native Kernels und inklusive Compile-Zeit belegen. JIT bleibt optional und experimentell, kein 1.0-Blocker.

### Arda 0.25 — Accelerated, Hybrid, and Distributed HPC

*Development codename: Ancalagon*

- **0.25.0:** Device API: allocation, transfer, event, synchronize, kernel capabilities, CPU-Backend als Referenz, GPU-Fehler an VM melden. CPU und GPU-Speicher nicht gleichsetzen; Unified Memory nicht als grenzenlos behandeln.
- **0.25.1:** Ein optionales GPU/iGPU-Backend (CUDA, Vulkan Compute, HIP oder anderes nach Zielhardware/Benchmarks), elementwise, reductions, GEMM und ausgewählte Sparse-Kernels, dtype capability checks; Float64 nicht auf jeder GPU sinnvoll schnell. Keine versteckten Kopien.
- **0.25.2:** Asynchrone Pipelines, CPU/iGPU/GPU-Hybridausführung, Device-Offloading, Partitionierung großer Arbeitslasten, Prefetch, ggf. MoE-expert placement. Datentransfer/Sync-Kosten im Plan. Weitere GPU-Backends bleiben optional; MPI wird über die folgenden HPC-Unterversionen ergänzt.
- **0.25.3:** MPI-Grundmodul aus 0.23 härten, Prozessgruppen, Kommunikations- und Synchronisationsvertrag, mehrknotige Tests mit Open MPI/MPICH. Nichtblockierende Transfers, sichere Request- und Buffer-Lebensdauer. Keine impliziten collectives aus GC.
- **0.25.4:** verteilte Dense-Tensoren/Vektoren, zeilenweise CSR, globale 64-Bit-Indizes, lokale 32-Bit-Indizes wo möglich, partitionierte Konstruktion ohne zentrale Vollmatrix. Halo-Pläne und Ghost-Werte, globale Norm/Reduktion, optional Block-Cyclic für Dense später. `dist` API v1.
- **0.25.5:** verteiltes SpMV, mindestens ein nativer validierter distributed CG/PCG für SPD, Domain Decomposition und Block-Jacobi; PETSc/hypre als optionale erweiterte Backends. Distributed Poisson mit globaler Residuenprüfung und Rank-Fehlerkoordination. GMRES/AMG/Parallel-FFT später abhängig von Bedarf.
- **0.25.6:** paralleles I/O via MPI-IO oder Parallel HDF5 optional, lokale Shards + globales Manifest, koordinierter atomarer Checkpoint, Checksums, Restart zunächst mit gleicher Partition; Repartition-Restart später. Kein Gather des vollständigen Outputs auf Rank 0.
- **0.25.7:** ACE-HPC: Topologie, Kommunikation, per-rank/per-node Budget, Thread/Ranks-Layout, Netzwerkprofil, lokale vs verteilte Pläne nur innerhalb zugewiesener Ressourcen. `getScaling`, `getCommunication`, `getTopology`, `getPartition`, `getEfficiency` mit Schätz-/Messlabel; Trace-Summaries, keine globalen Vollsnapshots.
- **0.25.8:** Slurm-Beispiele (srun, CPU-Binding, Aufgaben/Threads/RAM), echte Cluster-Tests, strong/weak scaling, Kommunikation, I/O und Lastbalance. CPU-only MPI-Pfad obligatorisch für den stabilen HPC-Release; GPU-aware MPI/NCCL, weitere GPUs und massive Skalierung optional/experimentell.
- **Gate:** Bei implementiertem GPU-Backend CPU/GPU-Vergleich innerhalb Toleranz, kontrolliertes Fehlverhalten ohne GPU, Benchmarks inkl. Transfers und Energie. CPU-only-Release bleibt vollständig nutzbar. HPC-Gate zusätzlich: verteiltes SpMV/PCG gegen serielle und unabhängige Referenz, globales Residuum, keine zentrale Vollmaterialisierung, reale Mehrknoten-Tests und dokumentierte Skalierung.

### Arda 0.26 — Machine Learning Inference and Training

*Development codename: Ainulindalë*

- **0.26.0:** NN-Modul: linear, conv, activation, normalization, softmax, loss, datasets, metrics, SGD/Adam/AdamW, model state/save/load, kleine Referenzmodelle (Regression, MLP, kleiner Transformer). Autodiff/AME/ACE wiederverwenden, Gradient Accumulation und Checkpointing standardmäßig verfügbar.
- **0.26.1:** Transformer-Inference: Embedding, RoPE, RMSNorm, Attention (GQA/MQA), MLP, Tokenizer und Sampling für zunächst ein klar definiertes kompatibles Modellformat/Architektur. Tiled attention/online softmax, quantisierte GEMV/GEMM, KV-Cache mit Budget, optional KV-Quantisierung und blockbasierte Caches. FlashAttention-Idee auf CPU-Cache anpassen; quadratische klassische Attention-Rechenarbeit bleibt.
- **0.26.2:** Out-of-core Inference: GGUF mapped weights, Layer/Expert Streaming, RAM-Cache für häufige Blöcke, SSD-Prefetch, Batch-/Prompt-Verarbeitung und I/O-Profiling. Optionale MoE-Unterstützung mit Expert-Routing, kein Versprechen, Dense-Modelle beliebig zu sparsifizieren. Speculative Decoding nur nach korrekter Sampling-Implementierung und gemessener Akzeptanz. 1B/7B zuerst, 70B disk-backed als Forschungsexperiment.
- **0.26.3:** Speicherarmes Fine-Tuning: LoRA, eingefrorene Gewichte, optional QLoRA, Gradient Accumulation, Activation Checkpoint/Offload, Adam8bit/Adafactor; experimentell GaLore, LOMO, MeZO, optimizer state offload. Backprop durch eingefrorene Basis weiterhin nötig; LoRA ≠ kostenloses Training. Trainierbare Gewichte und Basisgewichte getrennt bilanzieren.
- **0.26.4:** Scientific ML: Neural ODE, PINN, inverse Probleme, Surrogatmodelle; differenzierbare Simulationen mit validierten Gradienten. Kleine CPU-Modelle von Grund auf trainieren; große Modelle primär Inference/Fine-Tuning. Ternary/BitNet-spezialisierte Modelle später, nicht aus beliebigen FP64-Gewichten erzwingen.
- **0.26.5 — Distributed ML & reporting:** zunächst data parallel für kleine Modelle mit MPI-allreduce, korrekte Gradienten-Gewichtung, synchronisierte Optimiererzustände und Seed-Strategie; optional model/tensor/pipeline parallel, ZeRO-artige Sharding-/Offload-Verfahren und MoE Expert Placement als Forschung. `getTex` für Layer, Attention, Loss, Kettenregel; Run-Bericht für Parameter, Speicher, Kontext, T/s, Toleranzen und Trainingskurven. Keine vollständige Gewichts-/Gradientenaufzeichnung. GPU-Backends nicht für CPU-Training voraussetzen.
- **Gate:** Modell-/Tokenizer-Übereinstimmung gegen unabhängige Implementierung, Logit-/Qualitätsabweichungen, prefill vs decode T/s, TTFT, context length, exact encoding, RSS, I/O und Dauerleistung. CPU-Training kleiner Modelle reproduzierbar. LoRA-Parameter und Gradienten prüfen. 70B-Experiment nur mit Hardware-/Modell-/I/O-Angabe und ohne Mindesttokenrate-Versprechen. Kein vollständiges 70B-Pretraining als Laptopziel.

### Arda 0.27 — Scientific Developer Tools and Ecosystem

*Development codename: Imladris*

- **0.27.0:** CLI `arda run/repl/test/fmt/check/doc/build/profile`, bestehende CLI kompatibel halten. Formatter/Parser-Diagnose, Stacktraces, Source Locations.
- **0.27.1:** `arda.toml`, Lockfile, lokale/versionierte Pakete, Integritätsprüfsummen, reproduzierbare Abhängigkeiten, keine implizite Ausführung untrusted Install-Skripte. Öffentliches Register optional später.
- **0.27.2:** LSP, Syntaxhighlighting, Completion, Go-to-Definition, Debugger, Breakpoints, Inspektion, Jupyter-Kernel; Notebook-Sessions auf GC- und Ressourcenwachstum prüfen.
- **0.27.3:** CPU-/Memory-/I/O-Profiler, Graph-/Plan-Export, Benchmark-CLI, Explain-Ausgaben für Workspace und Budgetfehler; Dokumentation für Studenten (Tutorials ohne GPU) und Experten.
- **0.27.4:** `arda doc`, IDE/Notebook-Formelansicht, Assumptions/Units/Cost/Plan, verifizierte Berichte und Reproduzierbarkeitsmanifest. `compute.reproduce` prüft Inputs und Version, nicht automatisch untrusted Programme ausführen. Profiler für MPI, I/O, NUMA und Slurm-Jobvorlagen; keine heimlichen Clusterjobs. Elbisch/Englisch-Formattermodus, Dialekt-Tokenäquivalenztests, ASCII-Modulnamen und barrierearme wissenschaftliche Tutorials.
- **Gate:** Frische Ubuntu-Installation, Projekt anlegen, Daten importieren, wissenschaftliches Beispiel ausführen, testen, profilieren und exportieren ohne C-Änderungen.

### Arda 0.28 — Scientific and Performance Validation

*Development codename: Mandos*

- **0.28.0:** Plattformmatrix GCC/Clang, x86-64, ARM64, portable/BLAS/SIMD/threads/sanitize; optional macOS/Windows. Fuzzing (Lexer, Parser, Sparse, Dateien, Tensor-Shape), GC-Stress, Race Detection für Thread-Kernels. Lokale Testmatrix ab 0.14, hier vollständiger Audit. Keine GitHub Actions.
- **0.28.1:** Numerik: konditionierte/singuläre/random/analytische Fälle, Backward Error, Residuen, Solver-Convergence, ODE/PDE-Verfeinerung, Gradcheck, mixed precision, Quantisierungsqualität; Tests mit reproduzierbaren Seeds und problemabhängigen Toleranzen. Referenzen in ausreichend präziser Ausgabe/Binärformat, nicht bloß gerundeter Text.
- **0.28.2:** Benchmark-Matrix auf echten 8-/16-GB CPU-Laptops plus Workstation, optional ARM64 und GPU. Dauerlast, Energie (RAPL falls unterstützt), Cold/Warm Cache, harte RAM-Grenzen, I/O; fairer Vergleich gegen optimierte NumPy/SciPy/Julia, PETSc und llama.cpp je Anwendungsbereich.
- **0.28.3:** Externe Anwender-Pilotprojekte: Sparse FEM, 3D Poisson, 20-GB Messdaten, Low-Rank/Tensor-Train, differenzierbare Simulation, CPU-ML. Dokumentieren, welche Aufgaben einfacher, schneller oder ressourcenschonender sind, und wo Arda schlechter abschneidet.
- **0.28.4:** unabhängige AMDE-Validierung: golden LaTeX/MathML, mathematische Vergleichstests, Nichtausführung statischer Abfragen (Seiteneffekt-/Zufalls-/I/O-Sentinels), falsche Annahmen, dynamische Dispatch-Provenance, Trace-Abbruch/Limit und GC. Unit-/Uncertainty-/Interval-Referenzen, Reproduktion und Manifestprüfung. Overhead bei deaktivierten Zusatzfunktionen messen: Ziel <1 % große Kernels/Solver, <3 % kleine Calls/VM-Loops, Messrauschen und Startup/RSS separat; keine neuen Trace-Allokationen ohne Trace.
- **0.28.5:** reale MPI-Cluster-Validierung (mind. 2 physische Knoten für Mehrknotenaussage), 1/2/4/8 Ranks soweit verfügbar, strong/weak scaling, lokale RAM-/Kommunikationsstatistik, globales Residuum, gleiche Problemdefinition und Toleranz. Kein Cluster-Erfolg allein aus localhost ableiten. MPI-lokale Validierung kleine Jobs; auf Cluster optional größere Tests, Fehler-/Restart-Szenarien und dokumentierte Ausnahmen.
- **Gate:** Keine offenen kritischen Fehler in stabilen Kernkomponenten; dokumentierte Performance-Regressionsausnahmen; unabhängige reproduzierbare Ergebnisse. Experimentelle Features klar getrennt.

### Arda 0.29 — Release Candidate

*Development codename: Valinor*

- **0.29.0:** stabile API-Liste und Semantik-Freeze, Kompatibilitäts-/Migrationsleitfaden, öffentliche C-ABI soweit stabil, Dateiformatversionen, Fehlercodes und Ressourcenvertragsdokumentation. Experimentelle APIs separat kennzeichnen.
- **0.29.1:** Installationspakete, reproduzierbare Builds, Lizenz- und Dependency-Audit, Sicherheitsdokumentation, klare Nicht-Sandbox-Aussage, Tutorials und Referenzanwendungen.
- **0.29.x / 1.0.0-rc.N:** nur Stabilisierung, kritische Bugs, Doku und Regressionen. Kein großer Architekturwechsel im RC. Release aus sauberem Archiv testen.
- **Zusätzliches Gate:** stabile AMDE-Detailstufen und Provenance, numerische Verträge, reproduzierbare Berichte, minimale MPI/Distributed-API, Nutzerhandbuch für Laptop/Workstation/Cluster. Tolkien-Codenamen intern von neutralen APIs trennen; Marken-/Namens-/Lizenzprüfung vor öffentlichem Branding. Stabile und experimentelle Teilbereiche auch im gleichen Release klar ausweisen.


### Arda 1.0 — Stable Scientific Computing Language

*Development codename: Eä*

Pflicht: stabile Sprache/Runtime, Dense & Sparse, validierte zentrale lineare Algebra und Solver, portable CPU-Kernels, dokumentiertes Ressourcenmanagement, begrenzte Beyond-RAM-Operationen, Tensor-Autodiff mit Checkpointing, reproduzierbare Builds, öffentliche C-API, Beispiele/Dokumentation und klare Plattform-/Feature-Matrix. Grundlegendes CPU-ML soll demonstriert sein. Zusätzlich verbindlich: statisches getTex(0–3) für dokumentierte native und unterstützte Nutzerfunktionen, getCost/getPlan für unterstützte Operationen, Verifikations-API, begrenzte Traces, reproduzierbare Experimentberichte, optionale aber stabile grundlegende MPI-Unterstützung, verteilte CSR/Vektoren und mindestens ein validierter verteilter SPD-Solver mit Mehrknoten-Nachweis. Der Standardbuild bleibt ohne MPI/GPU-Abhängigkeit nutzbar. Fortgeschrittenes GPU, vollständiger JIT, 70B-Inference, TT-Solver, vollständiges QLoRA, allgemeines AMG und massiv verteiltes ML bleiben optional/experimentell und sind keine 1.0-Blocker. Falls reale Cluster-Validierung nicht möglich ist, HPC nicht als stabil behaupten; Release-Gate offen halten oder die Änderung des 1.0-Vertrags ausdrücklich beschließen. SemVer ab 1.0: Patch kompatible Korrekturen, Minor kompatible Features, Major inkompatible Änderungen.

## 5. Verbindliche Benchmark- und Ressourcenprofile

Referenz A: 8 GB RAM, 2–4 physische CPU-Kerne, keine dedizierte GPU, 2 GiB Zielbudget. Referenz B: 16 GB RAM, 4–8 physische CPU-Kerne, 6 GiB Zielbudget. Referenz C: Workstation, 32–128 GB RAM, variable CPU/optionale GPU. Das sind Zielprofile, noch keine gemessenen Referenzgeräte. Konkrete CPUs, RAM-Takt, SSD, OS, Compiler und thermische Bedingungen pro Test festhalten.

| ID | Ab Release | Workload | Ziel / Ergebnisprüfung |
|---|---|---|---|
| S1 | 0.14 | 1 Mio. Zeilen tridiagonal CSR, direkter Build+SpMV | Peak RSS <256 MiB; unabhängiger Matvec |
| S2 | 0.14 | SPD tridiag(-1,4,-1), n=1 Mio. | Peak RSS <512 MiB; relatives Residuum ≤1e-8; tatsächlicher Fehler |
| S3 | 0.14 | COO 3 Mio. unsortierte Einträge, Duplikate | Peak RSS und Zeit getrennt; kanonisches SciPy-Äquivalent; kein Dense |
| L1 | 0.15 | 100.000² low-rank, Rang 50 | direkte factorized Matvec, 80 MB Faktoren; Fehler dokumentiert |
| T1 | 0.16 | 8D Tensor Train, d=8, n=16, r=8 | 6.400 Float64-Werte =51.200 B Kerne; direkte Kontraktion, analytische Referenz |
| I1 | 0.16/0.21 | 20-GB echte Datei, Streaming-Statistik | Peak RSS <512 MiB; Referenz für mean/variance |
| I2 | 0.16/0.21 | 80-GB disk-backed Tensor, Reduction | Peak RSS <1 GiB; kein vollständiger RAM-Output |
| A1 | 0.17 | 1000 Schritte differenzierbare Simulation | Gradientenvergleich; Checkpoint-Zeit und Peak-RAM |
| K1 | 0.18 | 10 Mio. Elemente fused expression | keine 3 vollständigen temporären Float64-Arrays; Laufzeit vergleichen |
| K2 | 0.18 | GEMM/GEMV/SpMV, 1/2/4/8 Threads | sustained throughput, Roofline und Skalierung, Genauigkeit |
| P1 | 0.19 | 3D Poisson 128³ | Peak RSS <512 MiB, relatives Residuum ≤1e-8, Lösung validieren |
| F1 | 0.20 | 2D-FEM/Wärmeleitung | analytische oder unabhängige Referenz, Konvergenzordnung |
| I3 | 0.21 | disk-backed tiled GEMM | Input/Output > Budget, Fehler, I/O und Zeit, keine stillen Kopien |
| M1 | 0.26 | 1B/7B quantisierte CPU-Inference | TTFT, T/s, RSS, context, Modellqualität; keine pauschale Zielrate |
| M2 | 0.26 | 70B Q4 disk-backed | experimentell, 8–16 GB Host nach Eignung; I/O/Token und tatsächliche Rate |
| M3 | 0.26 | kleines CPU-ML-Training und LoRA | Loss/Konvergenz, Peak-RAM, Laufzeit, Qualität gegen Referenz |

### Methodik

- Compilerflags, Backend, Threadanzahl und Numerikmodus festhalten; dieselben Daten, Shape, Präzision, Solver-Toleranz und Ausgabeanforderungen für Konkurrenten. Keine Vergleiche zwischen Python-Schleifen und optimiertem C als Beleg gegen NumPy.
- Separat: Startzeit, Compile-/Warmup-Zeit, Median aus mindestens 5–9 geeigneten Wiederholungen und Streuung; für lange Workloads weniger Wiederholungen mit dokumentierter Begründung. Cold/Warm-Cache sowie sustained performance getrennt. Kein unkontrolliertes Cache-Dropping auf fremden Systemen.
- Linux: `/usr/bin/time -v`, `perf stat` (falls verfügbar), `/proc` I/O, cgroup v2 memory.peak/memory.events, RAPL sofern unterstützt. Interne Bytezähler und externe RSS/Cache-Zähler getrennt. I/O einschließlich Prefetch und Writeback erfassen.
- Performance-Gate: auf fixer Referenzhardware unerklärte >10% Median-Regression in stabilen Kernworkloads untersuchen; Schwelle ist Ziel, kein universeller Beweis. Sicherheits-/Korrektheitsfixes dürfen begründete Regressionen verursachen. Keine übertragbaren Laufzeitbehauptungen ohne Messung.
- Accuracy-Gate: je Algorithmus absolute/relative Toleranz, Rückwärtsfehler, Konvergenz und ggf. Modellqualität; deterministischer Modus dokumentieren. Quantisierung/Low-Rank als ausdrücklich erlaubte Approximationen testen.

## 6. Kostenmodelle und konkrete Grenzen

- CSR-Speicher auf 64-bit mit uint32-Spalten: 12*nnz + 8*(rows+1) Byte. COO: 16*nnz. COO ist nicht immer größer als CSR (viele leere Zeilen können offsets dominieren). Dense: 8*rows*cols für Float64. Builder/Sort/Conversion-Peak enthält gleichzeitig alte und neue Daten und Workspace.
- Matrixfreie 128³ Poisson: 2.097.152 Unbekannte, ein Float64-Vektor 16 MiB, sechs Vektoren 96 MiB. Solver-Arbeitsaufwand hängt von Konditionierung ab; feines Poisson mit CG ohne Multigrid kann viele Iterationen benötigen.
- TT (d=8, n=16, uniform internal rank r=8): dense 16^8=4.294.967.296 Werte, 34.359.738.368 B =32 GiB. TT-Kerne 2*n*r+(d-2)*n*r²=6400 Werte, 51.200 B=50 KiB. Nur wenn diese Ränge mathematisch ausreichen; direkte Konstruktion und Kontraktion nötig.
- 70B FP16: 140 GB Gewichte; ideal Q4 ohne Metadaten 35 GB, Beispiel Q4 mit 32 Werten/4 B scale 43,75 GB. KV-Cache extra; Llama-artige 80-Layer, 8 KV Heads, HeadDim 128, 4096 Token FP16: 1,25 GiB. Bei 40 GB SSD-Lesen pro Token und 3 GB/s maximal 0,075 Token/s allein durch I/O. 40 GB aus RAM bei 50 GB/s ideal maximal 1,25 Token/s allein durch Speichertraffic. Beide ohne Rechenzeit/andere Daten. Cache, MoE und Batching ändern die tatsächlich gelesene Bytezahl.
- LoRA bei 4096², Rang16: 131.072 trainierbare Werte statt 16.777.216, 0,78125%; die eingefrorene Basis bleibt notwendig. Standardbeispiel Mixed Precision Adam mit 16 B/Parameter: 7B →112 GB, 70B →1,12 TB vor Aktivierungen. 70B mit 1 Mio. Trainingstokens und idealisierten 100 GFLOP/s bei 6ND ≈48,6 Tage; echtes Pretraining nutzt viel mehr Tokens.
- Roofline: P ≤ min(P_peak, RAM bandwidth * arithmetic intensity), zusätzliche SSD-Grenze bei disk-backed workload. Für blockweise Ausführung mit idealer Überlappung ist die langsamste Pipeline-Stufe limitierend; bei fehlender Überlappung addieren sich Zeiten. Reale Planung über kalibrierte Messdaten, nicht nur theoretische Peaks.

## 7. Abhängigkeiten und Priorisierung

0.14 sparse+budget → 0.15 structured math → 0.16 storage/dtype → 0.17 tensor AD → 0.18 kernel/ACE. 0.14/15/16 → 0.19 operator solvers → 0.20 simulation. 0.16/18 → 0.21 disk execution; 0.16/17/18/21 → 0.22 planner; 0.16/22 → 0.23 API and model formats; 0.18/22 → 0.24 compiler; 0.16/18/23 → 0.25 devices; 0.16/17/18/21/23 → 0.26 ML. ML CPU inference does not require GPU/JIT. Small ML proofs and laptop benchmarks should be developed continuously before 0.26.

P0: sparse, accounting, safe allocation/ownership, matrix-free, validated math, basic mmap/chunked, portable CPU performance. P1: AD/checkpoint, kernels/fusion, threads, multigrid, mixed precision, beyond-RAM scheduler, low-rank. P2: specialized ML/TT, advanced simulation, JIT and GPU as appropriate. P1 (zusätzlich): AMDE core, Verify, MPI-Grundlagen und verteilter SPD-Solver als 1.0-Ziele. P2 (zusätzlich): Units, Uncertainty, Interval, erweiterte Symbolik und zusätzliche HPC-Backends. P3: 70B-offload experiments, ternary, large distributed training, general TT solvers and ambitious hardware backends. Priorities may be revised after benchmark evidence; optional projects do not block stable core releases.

## 8. Definition of Done — every feature/release

1. API/semantics, dtype/shape, mathematical assumptions, error conditions and memory bounds written before coding.
2. Portable implementation or explicitly optional backend; readable code, checked size/index arithmetic and GC/ownership/cleanup reviewed.
3. Unit, property, edge, invalid input, OOM/budget, GC-stress tests; independent reference tests for numerical features. Sanitizers and existing regression tests pass.
4. Benchmarks against prior release and relevant optimized competitor; distinguish model estimates and measured values; report no-benefit cases too.
5. Example program, API docs, limitations, changelog, exact reproducible build commands; no silent feature activation reducing precision.
6. Versioned source archive or git tag, clean build from final artifact, optional build variants tested; no unverified claims about platforms.
7. AMDE-Descriptor und mathematische Voraussetzungen synchron zur Implementierung aktualisieren; optionale Funktionen deaktiviert benchmarken; bei verteilten Features Rank-Fehler, kollektive Reihenfolge, Partitionen und Kommunikation prüfen.
8. Experimental/stable feature labels and migration notes; open issues recorded. Failed release gate prevents claiming stable implementation.

Suggested lokale Validierung targets: `make test`, `make test-reference`, `make sanitize`, `make check-generated`, `make format-check` (current targets); later `make test-sparse`, `make test-memory`, `make bench-laptop`, `make bench-beyond-ram`, `make fuzz`, `make test-threads` (proposals). Use fixed seeds and small lokale Validierung workloads; large data tests opt-in on suitably provisioned machines.

## 9. Sparse Computing Implementation Order

| Work package | Deliverable | Depends on |
|---|---|---|
| S0 | Reproduce baseline; inspect all OType switches/native registration and error paths; lokale Validierung setup | 0.13.1 |
| S1 | Sparse format/API/error/memory spec, mathematical descriptor contract, local/global index design, test vectors and checked arithmetic | S0 |
| S2 | Native object type, accounting, allocation and release; GC-stress | S1 |
| S3 | COO constructor and inspection; validation tests | S2 |
| S4 | Canonical CSR/CRS constructor and inspection; offsets/indices tests | S2 |
| S5 | Bidirectional conversions, duplicate and zero handling, dense bridges | S3/S4 |
| S6 | Typed builder, tridiagonal constructors, memory budget and conversion benchmarks | S5 |
| S7 | SpMV, transpose, scale/add, SpMM, Matrix Market | S5/S6 |
| S8 | Operator contract, PCG and workspace | S7 |
| S9 | SciPy comparisons, sanitizer, million-row benchmark, examples and release archive | S0–S8 |

0.14.0 implements S0–S5. Milestones 0.14.1–0.14.3 implement S6–S8. Arda 0.14.4 completes S9. [Release validation](docs/MORIA_0.14.4.md) Scope creep into full tensor rewrite, JIT or transformer engines is explicitly excluded. Update this roadmap and version-specific specs when evidence changes implementation choices.

## 10. Zusätzliche Abnahmebenchmarks: AMDE, Numerik, HPC

Die folgenden Ziele ergänzen die ursprüngliche Benchmark-Tabelle. Keine gemessenen Werte. Benchmarks mit gleichwertigen Algorithmen, Daten, Toleranzen, Threadzahlen, I/O- und Hardwarebedingungen; unabhängige Referenzen einschließlich Handcalcs/SymPy für Darstellungsfragen und PETSc für verteilte Solver. Peak-RSS pro Prozess vs. aggregierter Node-Speicher und cgroup-Accounting unterscheiden.

| ID | Release | Test und Abnahmekriterium |
|---|---|---|
| D1 | 0.15 | getTex(0–3) auf einfachen Funktionen und nativen Algorithmen; kein Funktionsaufruf oder Seiteneffekt; goldene Formeln, korrekte Klammerung, bekannte/unbekannte Pfade |
| D2 | 0.15/22 | getCost/getPlan auf symbolischen Shape-Spezifikationen, Abweichung Schätzung vs. Messung dokumentieren, Budget einschl. Output und Workspace |
| D3 | 0.19 | 1 Mio. Unbekannte, kompakter CG-Trace; Ziel Tracebudget 16 MiB inkl. Overhead, keine vollen Vektor-Kopien, Kürzung sichtbar |
| D4 | 0.18/28 | Trace/Units/AD ungenutzt: Ziel <1 % Kernel-/Solver-Overhead, <3 % VM-Overhead, zusätzlich Compile-Zeit/Binary/RSS; statistisch belastbar messen |
| V1 | 0.19 | Verifikation gut/schlecht konditionierter Systeme, tatsächlicher Rückwärtsfehler, Nichtkonvergenz und NaN/Inf |
| V2 | 0.16/17 | Unit-Dimensionen und affine Temperatur, AD-Unsicherheitsfortpflanzung, HVP gegen kleine volle Hessian |
| R1 | 0.20/27 | Checkpoint/Restart eines ausgewählten ODE-Solvers, Daten-/Versionsfehler erkennen, toleranzäquivalentes Resultat |
| R2 | 0.27/28 | Experiment auf zweitem Rechner anhand Manifest reproduzieren, bitweise/numerisch/statistisch unterscheiden |
| H1 | 0.18 | STREAM-artige Bandbreite, 1/2/4/8 Threads, NUMA first-touch wo vorhanden, BLAS-Oversubscription vermeiden |
| H2 | 0.23 | MPI Ping-Pong und collectives, 1/2/4 Ranks auf einem Host, Lifetime-/Fehlerfälle |
| H3 | 0.25 | Distributed SpMV und SPD-PCG: globales Residuum ≤1e-8, Referenzlösung, keine volle Matrix/Output-Replikation, Speicher pro Rank |
| H4 | 0.25/28 | Strong scaling bei fester Problemgröße, weak scaling bei fester Größe pro Rank, mindestens 2 reale Nodes; Speedup/Effizienz/Kommunikationsanteil berichten |
| H5 | 0.25 | MPI-IO/partitionierte Checkpoints: Restart-Integrität, per-node RAM, Schreib-/Lesedurchsatz |
| H6 | 0.26 | Kleine CPU data-parallel ML-Aufgabe: Gradient/Loss gegen serielle Referenz, Durchsatz und Allreduce-Kosten |

Referenz D: HPC-System mit mindestens zwei Knoten (konkrete Ausstattung erst bei Zugang), feste Rank/Thread-Zuordnung, Slurm oder vergleichbare Scheduler. Benchmark-Skalierung nicht mit reinem lokalen Multithreading verwechseln. Strong scaling S(p)=T(1)/T(p), Effizienz S(p)/p; weak scaling E(p)=T(1,n)/T(p,pn). Bei unterschiedlichen Knotenkonfigurationen die Baseline angeben. Amdahl bei 99 % parallel und 64 gleichartigen Recheneinheiten ≈39,3× idealisiert; Netzwerk kommt hinzu. Beispiel verteilter Float64-Vektor mit 20 Mrd. Elementen: 160 GB global, 20 GB pro Rank bei 8 gleich großen Partitionen, vor Workspaces/Halo. HPCG/LINPACK/STREAM als externe Referenzmethoden oder klar gekennzeichnete eigene ähnliche Workloads, keine unberechtigte Benchmark-Zertifizierung.

## 11. Projektidentität und Entwicklungscodenamen

Arda ist eine unabhängige, MIT-lizenzierte wissenschaftliche Programmiersprache. Technische Meilensteine und mathematische APIs stehen im Vordergrund. Die optionalen, an Quenya und Sindarin angelehnten Schlüsselwörter und die literarischen Entwicklungscodenamen bleiben Teil der Projektkultur. Die Projektzuordnung ist im [README](README.md#license-and-attribution) erläutert.

### 11.1 Aktuelle Syntax (0.13.1, lexer.c)

| Elbisch inspiriert | Englisches Alias | Semantik |
|---|---|---|
| esta | let | Deklaration |
| caro | fn / def | Funktion |
| nan | return | Rückgabe |
| qui | if | Bedingung |
| ai | else | Alternative |
| tenna | while | Schleife |
| pedo | print | Ausgabe |
| nae | true | Wahr |
| law | false | Falsch |
| mun | nil | Nil |
| an | for | Iteration |
| nost | class | Klasse |
| tol | import | Import |

13 inspirierte Schreibweisen, 13 Tokenarten mit solchen Alternativen. Beide Dialekte werden auf dieselben internen Tokens abgebildet, keine unterschiedlichen mathematischen Semantiken oder Kernels. Neue Sprachfeatures zunächst mit klaren englischen Keywords; elbische Aliase optional nach Prüfung von Lesbarkeit, Kollisionen und Herkunft. `getTex`, `solve`, `matmul`, `mpi` und ähnliche APIs bleiben verständlich. Intern Unicode-Codenamen, technische Dateipfade/Module ASCII (z. B. rumil), falls die Plattform dies erleichtert. Tests für gemischte Dialekte, Formatter mit expliziter Dialektwahl, englische Doku und optional Tolkien-Beispiele. Keine Übernahme geschützter Illustrationen/Filmlogos als Branding.

### 11.2 Historische Releases (aus CHANGELOG.md)

| Version | Tatsächlicher Titel |
|---|---|
| 0.6 | The Eldar VM |
| 0.7 | Shadowfax |
| 0.8 | Rivendell |
| 0.8.1 | Shadowfax Performance Update |
| 0.9 | Ithil Scientific |
| 0.10 | Anor |
| 0.11 | Narya |
| 0.12 | Nenya |
| 0.13 | High-dimensional scientific computing |
| 0.13.1 | Scientific maintenance |

0.13/0.13.1 werden historisch nicht rückwirkend umbenannt. Vilya setzt später die Narya/Nenya-Reihe fort.

### 11.3 Vorgeschlagene Codenamen bis 1.0

| Version | Technischer Schwerpunkt | Entwicklungscodename | Tolkien-Bezug / Metapher |
|---|---|---|---|
| 0.14 | Sparse, Aufbau und Speicherfundament | Moria | Unterirdische Hallen und strukturierter Aufbau |
| 0.15 | Scientific Foundations, MathIR, getTex | Rúmil | Gelehrter von Tirion und frühes Schriftsystem Sarati |
| 0.16 | Virtuelle Tensoren, kompakte Speicherformate | Mithril | Leichtes, widerstandsfähiges Metall |
| 0.17 | Autodiff, Sensitivität, Checkpointing | Vilya | Dritter der Elbenringe; Fortsetzung Narya/Nenya |
| 0.18 | Maximale CPU-Performance | Shadowfax II | Gandalfs schnelles Pferd; Fortsetzung der Performance-Releases |
| 0.19 | Numerische Solver | Aulë | Vala des Handwerks und der Gestaltung |
| 0.20 | Physikalische Simulationen | Ulmo | Vala der Gewässer |
| 0.21 | Beyond-RAM Datenverarbeitung | Khazad-dûm | Dieselbe Zwergenstadt wie Moria; hier bewusst Rückbezug auf 0.14 |
| 0.22 | Adaptive Planung und Introspektion | Palantír | Sehende Steine als Metapher für Sicht auf den Ausführungsplan |
| 0.23 | Interoperabilität und MPI-Grundlagen | Eärendil | Reisender und Vermittler zwischen Welten |
| 0.24 | Optimierender Compiler | Fëanor | Handwerk, Erfindung und Tengwar |
| 0.25 | GPU, hybride und verteilte HPC-Systeme | Ancalagon | Gewaltiger Drache als Metapher für große Rechenleistung |
| 0.26 | ML, Training und Inference | Ainulindalë | Musik der Ainur als Metapher für viele zusammenwirkende Teile |
| 0.27 | Entwicklerökosystem | Imladris | Elbischer Name Bruchtals; Ort des Wissensaustauschs |
| 0.28 | Validierung, Prüfung, Audit | Mandos | Hallen und Urteil als kreative Metapher für Prüfung |
| 0.29 | Release Candidate | Valinor | Land der Valar als Ziel der langen Reise |
| 1.0 | Stabile wissenschaftliche Runtime | Eä | Die erschaffene Welt/Universum als Metapher für das Gesamtwerk |

Rúmil bleibt der interne Name für AMDE, auch nach 0.15. `mathdoc` ist der neutrale öffentliche Modulname, `rumil` ein möglicher Alias. Module `compute`, `sparse`, `linalg`, `units`, `uncertainty`, `mpi`, `distributed`, `nn` behalten beschreibende Namen. Referenzen für die Namensherkunft: [Rúmil](https://tolkiengateway.net/wiki/Rumil_of_Tirion), [Moria/Khazad-dûm](https://tolkiengateway.net/wiki/Moria), [Vilya](https://tolkiengateway.net/wiki/Vilya), [Aulë](https://tolkiengateway.net/wiki/Aul%C3%AB); übrige Namen als kreative Vorschläge. Tolkien-Primärwerke als kultureller Hintergrund: The Silmarillion und The Lord of the Rings. Vor Veröffentlichung Namens-/Markenkollisionen prüfen (auch den Namen Arda selbst); MIT-Lizenz des Codes klärt keine fremden Markenrechte.

## 12. Abhängigkeiten, Release-Entscheidungen und Scope

Kritischer Pfad: 0.14 Sparse + Allokation → 0.15 Algorithmen/Deskriptoren → 0.16 Buffer/Storage/DType → 0.17 AD und 0.18 AKE → 0.19 Operatoren → 0.20 Simulation / 0.21 Out-of-Core → 0.22 ACE/Verträge → 0.23 C-API/MPI → 0.25 Distributed HPC → 0.28 Validierung → 1.0. 0.24 JIT kann unabhängig experimentell laufen und ist kein stabiler Kern-Blocker. AMDE entwickelt sich parallel ab 0.14, MPI-Design ab 0.14, Implementierung ab 0.23. ML nutzt bestehende Storage-, AD- und Kernel-Infrastruktur statt neuer isolierter Engine. Externe Nutzer können ab 0.15/0.18 mit Teilfunktionen arbeiten; kein Anspruch, vor 1.0 sämtliche APIs stabil zu halten.

Verbindlicher Kern: sichere Runtime, native Sparse/Dense, portabler optimierter CPU-Pfad, ausgewählte Out-of-Core-Operationen, Tensor-AD, validierte Solver, AMDE/Verify, Ressourcenbudgets, C-API, MPI-Grundmodul und verteiltes CSR/PCG. Erweiterungen mit eigenem Gate: GPU, Advanced ML, Units/Uncertainty/Interval, symbolische Mathematik, FEM, paralleles I/O, JIT. Forschungsziele ohne 1.0-Blockade: 70B-SSD-Inference, QLoRA sehr großer Modelle, Ternary, allgemeine TT-Solver, vollständige Cluster-Automatik, massive GPU-Skalierung, umfangreiches CAS. Keine feste Zeitplanung; ambitionierter Mehrjahresplan, Umfang nach Messung und verfügbaren Entwicklern neu priorisieren. Wissenschaftliche Korrektheit und überprüfbare Release-Qualität dürfen nicht für Versionsnummern geopfert werden.

## 13. Gemeinsame Ziel-API (Illustration, nicht implementiert)

```python
tol sparse as sp
tol compute
tol mathdoc

compute.configure({"device": "cpu", "threads": 4,
                   "memory_limit": "2 GiB", "allow_offload": nae,
                   "allow_approximation": law})

caro system(A, b):
    nan sp.cg(A, b)

# Statisch: keine Berechnung starten
pedo(system.getTex(3))
pedo(system.getCost({"shape": [1000000, 1000000], "nnz": 2999998}))

# Numerik, optionaler Trace und Verifikation
esta A = sp.tridiag(1000000, -1.0, 4.0, -1.0)
# b wird problemabhängig erzeugt
# run = system.trace([A, b], {"max_memory": "16 MiB", "sample_every": 10})
# mathdoc.export(run, "report.tex")
```

Schnittstellen für Dot-Methoden, Keyword-Argumente und Konfiguration werden im jeweiligen Release spezifiziert. Trace-API soll das Ergebnis desselben aufgezeichneten Runs zurückgeben, nicht die Berechnung erneut für die Dokumentation ausführen. Distributed-Varianten explizit konstruieren und unter zugewiesenen MPI-Ranks starten. Gleiche mathematische API soweit unterstützt, keine magische Parallelisierung beliebiger dynamischer Programme.

## 14. Technical references for later design reviews

These references are technical starting points from the prior research, not claims that Arda already implements their techniques.

- Roofline: https://doi.org/10.1145/1498765.1498785
- Tensor Train: https://doi.org/10.1137/090752286
- Randomized SVD: https://doi.org/10.1137/090771806
- PETSc: https://petsc.org/
- FlashAttention: https://arxiv.org/abs/2205.14135
- FlexGen: https://arxiv.org/abs/2303.06865
- LLM in a Flash: https://arxiv.org/abs/2312.11514
- PowerInfer: https://arxiv.org/abs/2312.12456
- llama.cpp: https://github.com/ggml-org/llama.cpp
- BitNet: https://github.com/microsoft/BitNet
- KIVI: https://arxiv.org/abs/2402.02750
- Checkpointing: https://arxiv.org/abs/1604.06174
- LoRA: https://arxiv.org/abs/2106.09685
- QLoRA: https://arxiv.org/abs/2305.14314
- 8-bit optimizers: https://arxiv.org/abs/2110.02861
- GaLore: https://arxiv.org/abs/2403.03507
- MeZO: https://arxiv.org/abs/2305.17333
- ZeRO-Infinity: https://arxiv.org/abs/2104.07857

- MPI Forum (MPI standard): https://www.mpi-forum.org/docs/
- Open MPI: https://www.open-mpi.org/
- MPICH: https://www.mpich.org/
- hwloc: https://www.open-mpi.org/projects/hwloc/
- Slurm: https://slurm.schedmd.com/documentation.html
- Handcalcs: https://github.com/connorferster/handcalcs
- Handcalcs.jl: https://github.com/co1emi11er2/Handcalcs.jl
- SymPy: https://docs.sympy.org/
- Symbolics.jl: https://symbolics.juliasymbolics.org/
- Pint: https://pint.readthedocs.io/
- MPFR: https://www.mpfr.org/

**Leitmotiv:** Compute. Optimize. Understand. — Advanced science on everyday hardware, scalable to HPC. Maximum practical performance, transparent resource use and validated numerical results.

### Repository policy — local testing only

No GitHub Actions workflows or hosted CI jobs are enabled or planned under the current project policy. Run the existing Makefile test, reference-test and sanitizer targets locally. Cross-platform and HPC validation may be run on explicitly available local or cluster hardware. This policy supersedes any remaining references to hosted CI in historical documents.
