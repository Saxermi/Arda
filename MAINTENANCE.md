# Arda 0.13.1 — Entwicklungsbericht

Basis: hochgeladenes Archiv `arda-0.13-scientific.zip`. Dieser Release verändert keine numerischen Algorithmen und enthält noch kein Sparse-Modul. Das COO/CSR-Konzept steht in ROADMAP_0.14.md.

## Änderungen

- **Formatierung:** Die bisherigen 13 C-Dateien und `arda.h` wurden token-erhaltend umformatiert; Kommentare und Einrückung wurden ergänzt. clang-format war in der Testumgebung nicht installiert und konnte dort nicht heruntergeladen werden. Eine `.clang-format`-Konfiguration und `make format` stehen für die spätere Standardisierung zur Verfügung. Der Vergleich der C-Tokens vor/nach der Umformatierung (ohne Kommentare) war identisch. Generierte Header werden nicht manuell formatiert.
- **Frontend:** `frontend.c`/`frontend.h` implementieren `prepare_source` und eine verkettete Liste temporärer Allokationen für Lexer, Parser und Compiler. Bei einem Syntax-/Compilerfehler springt die Fehlerbehandlung zum Frontend-Scope zurück und gibt sämtliche dort noch registrierten Allokationen frei. Bei Erfolg werden AST und Tokens freigegeben und die Bytecode-Allokationen aus dem Scope gelöst. Die Allokationsheader bleiben bis zur Freigabe erhalten. OOM bleibt fatal; keine Sandbox und kein Schutz vor allen bösartigen Eingaben.
- **REPL/Import:** `main.c` nutzt `prepare_source`. Syntaxfehler beenden die REPL nicht mehr; der bisherige globale Zustand bleibt erhalten. Fehlerhafte Dateiskripte behalten Exitcode 1. `vm.c` übersetzt Frontend-Fehler bei Importen in abfangbare Laufzeitfehler. Fehlende Eingabedateien werden sauber behandelt. Keine Änderung an der bisherigen Blockeingabe-Logik.
- **Bytecode-GC:** `Proto.root` verbindet Kindfunktionen mit dem Wurzelprogramm; `Proto.marked` dient der GC-Markierung. `gc.c` markiert Wurzelprogramme über Frames, erreichbare Funktionsobjekte und Parameterumgebungen. Nach dem Objektsweep werden unmarkierte Programmgruppen freigegeben. Das ist gruppenweises, kein funktionsweises Freigeben; eine überlebende Closure hält ihre gesamte Programmgruppe am Leben. Der Programmzeiger-Vektor kann seine bisherige Kapazität behalten. `--gc-stats` meldet `program_roots` nach einer abschließenden Sammlung; `bytes` ist weiterhin die bisherige Objekt-Speicherstatistik, keine vollständige Bytecode- oder Prozess-RAM-Messung.
- **Sanitizer:** `make sanitize` setzt ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 und UBSAN_OPTIONS=halt_on_error=1. Die vorherige Deaktivierung der Leak-Erkennung entfällt. Sechs Regressionstestmethoden ergänzen die vorhandenen 120 Tests. Darunter 100 Wiederholungen dreier fehlerhafter Frontend-Eingaben und 3.000 REPL-Zuweisungen, Closure-Lebensdauer, Importfehler und CLI-Fehler.
- **Referenztests:** `tests/reference_numerics.py` enthält acht optionale Testmethoden mit NumPy/SciPy, fester Seed 1301. Größere Tensoren werden über `nc.tolist` ausgegeben, damit die Anzeige sie nicht abkürzt; verglichen werden die ausgegebenen Zahlen (15 signifikante Stellen), nicht die rohen Float64-Bits. Toleranzen hängen vom Verfahren ab. Dies ist eine Stichprobenvalidierung, keine allgemeine numerische Korrektheitsgarantie.
- **Build:** Separate Verzeichnisse `build/blas0-lapack0-sanitize0`, `build/blas1-lapack1-sanitize0`, `build/blas0-lapack0-sanitize1` usw. `.o` und automatisch erzeugte `.d` pro C-Datei (`-MMD -MP`); Flag-Stamp bei Änderung der Build-Parameter; `./arda` ist ein Symlink zum zuletzt ausgewählten Build. Tests verwenden den absoluten `ARDA_BIN`-Pfad. `make clean` löscht alle Konfigurationen. Generatoren für `ode` und `opt` teilen `tools/embed_module.py`, bewahren UTF-8-Bytes und schreiben atomar. Änderungen an `lib/*.arda` lösen automatisch die Regenerierung des zugehörigen Headers und den Neubau von `vm.o` aus. `make check-generated` prüft die Konsistenz.

## Gemessene Validierung

Umgebung: Linux Debian 13, GCC 14.2, Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0. Kein Test auf Ubuntu oder ARM64 in dieser Umgebung. Der bestehende GCC-Sanitizer-Hinweis `-Wclobbered` zu `self` in `vm.c` erschien bereits in 0.13.0. Keine Sanitizer-Fehler in den ausgeführten Tests.

| Build | Reguläre Tests | Referenztests |
|---|---:|---:|
| Standard | 126/126 | 8/8 |
| BLAS + LAPACK | 126/126 | 8/8 |
| ASan + UBSan + Leak-Erkennung | 126/126 | 8/8 |

Die alte Version wurde zum Vergleich ausschließlich mit einer Diagnoseausgabe für `v->np` kompiliert. 3.002 REPL-Eingaben (Deklaration, 3.000 Zuweisungen, Ausgabe) ergaben in beiden Versionen 3000. Alt: 3002 gespeicherte Programmgruppen; neu: 0 nach GC. Eine Closure hält weiterhin ihre Programmgruppe lebendig (1), nach Entfernen der letzten Referenz und GC: 0.

| Referenzvergleich (Standard-Build) | Maximaler absoluter Fehler |
|---|---:|
| 20 zufällig erzeugte solve-Systeme (n=2,4,8,16,32) | 5.55e-16 |
| 5 Matrixmultiplikationsformen | 5.33e-14 |
| Skalarprodukt, 1001 Elemente | 1.42e-14 |
| 8 elementweise/Reduktions-/Broadcast-Ergebnisse | 4.88e-15 |
| FFT/IFFT, 1 bis 1024 Elemente | 9.91e-13 |
| CG | 7.88e-14 |
| CGLS | 6.94e-16 |
| Poisson (5 unabhängige Sparse-Referenzoperatoren, bis 8D) | 2.63e-12 |
| RK45 gegen SciPy solve_ivp | 4.77e-12 |
| L-BFGS Rosenbrock gegen SciPy BFGS | 2.05e-10 |

Zusätzlich ein solve-Pivot-Test mit A=[[0,2],[1,3]], b=[4,7], x=[1,2]. Simpson auf sin(x) von 0 bis pi (100 Intervalle): 2.0000000108245; Trapezregel: 1.99983550388744 (exakter Wert 2). Diese Fehler stammen aus Stichproben, nicht aus einer theoretischen Fehlergrenze.

Inkrementeller Build: Änderung von scientific.c → 1 C-Kompilierung + 1 Link; unmittelbar folgender Build ohne Änderung → 0 C-Kompilierungen. Eine vorübergehende Kommentaränderung in `lib/ode.arda` wurde automatisch in `src/ode_source.h` übernommen; nur `vm.o` wurde neu kompiliert. Anschließend wurde die ursprüngliche Moduldatei wiederhergestellt und `make check-generated` bestanden.

## Reproduzieren

```sh
make -j4 test
make sanitize
make BLAS=1 LAPACK=1 test
make check-generated
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-dev.txt
make test-reference
make BLAS=1 LAPACK=1 test-reference
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 make SANITIZE=1 test-reference
make
./arda --gc-stats --gc-stress examples/highdim.arda
```

Alte Benchmark-Dateien im Archiv sind historische Messungen für frühere Releases, keine neuen 0.13.1-Benchmarks. CI und Sprachspezifikation bleiben nach dem angeforderten Umfang offen. COO und CSR folgen erst in 0.14.
