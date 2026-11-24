# Historical documentation (German)

This is the original documentation shipped with 0.13.1, preserved for historical context. Earlier release sections contain superseded limitations (notably REPL recovery, bytecode GC, sanitizer configuration and adaptive RK45); consult the current English README and MAINTENANCE.md for the actual current state. No hosted CI is planned under the current repository policy.

---

# Arda 0.13.1 — Scientific maintenance

Arda ist eine kleine, dynamisch typisierte, interpretierte Programmiersprache mit Python-ähnlicher Einrückung und von Quenya/Sindarin inspirierten Schlüsselwörtern. Die Begriffe sind künstlerisch inspiriert, keine verbindlichen Übersetzungen. Diese Version baut auf 0.9 Scientific auf und ergänzt Tensoren, numerische Verfahren und automatische Differentiation.

**Compiler und Runtime sind in C11 implementiert.** Das allgemeine ODE-Modul ist in Arda selbst geschrieben und in die Runtime eingebettet. Python wird für die automatischen Header-Generatoren, Tests und Benchmarks benötigt, nicht zur Ausführung von Arda-Programmen. Arda ist ein experimenteller Sprachprototyp, keine produktionsreife oder sandboxed Runtime.

## Installation (Ubuntu / Linux)

```sh
sudo apt install build-essential python3 unzip
unzip arda-0.13.1-maintenance.zip
cd arda-0131
make
./arda --version
./arda examples/fellowship.arda
./arda examples/rings.arda
make test
```

Optional: `sudo install -m 755 arda /usr/local/bin/arda`.

`make sanitize` baut mit AddressSanitizer, UndefinedBehaviorSanitizer und aktivierter Leak-Erkennung. `make` wechselt anschließend ohne `make clean` zurück zum Standard-Build. `make test-reference` führt unabhängige Referenztests mit NumPy und SciPy aus (`python3 -m pip install -r requirements-dev.txt` in einem virtuellen Python-Environment). `make check-generated` prüft eingebettete Module. `--gc-stats` zeigt unter anderem die Zahl lebender Bytecode-Programmgruppen. Details: [MAINTENANCE.md](MAINTENANCE.md); Planung für [0.14](ROADMAP_0.14.md).

## Scientific Computing: Grundlagen (seit 0.9)

`tol numc as nc` importiert das in C implementierte numerische Modul. Arrays sind
GC-verwaltete Float64-Puffer (8 Byte pro Element, row-major); Views können nicht zusammenhängend sein.
Ein Build ohne weitere Bibliotheken ist Standard; für CBLAS-Matrixmultiplikation
optional unter Ubuntu `sudo apt install libblas-dev` und `make BLAS=1`.
Mit `nc.backend()` lässt sich der aktive Backendname abfragen.

```arda
tol numc as nc
esta x = nc.linspace(0, 4, 5)
esta y = 2 * x + 1
pedo(y)
pedo(nc.mean(y))
esta A = nc.array([[3, 2], [1, 2]])
pedo(nc.solve(A, nc.array([7, 5])))
```

| API | Funktion |
|---|---|
| array(list), zeros(n), ones(n) | Vektoren und Tensoren aus rechteckigen verschachtelten Listen |
| zeros(r,c), ones(r,c), linspace(a,b,n) | Tensor (1–8 Dimensionen) oder äquidistantes Gitter inkl. Endpunkten |
| shape(a), bytes(a), tolist(a), backend() | Metadaten, Konvertierung und Build-Backend |
| sum(a), mean(a), var(a), std(a), min(a), max(a) | Reduktionen über alle Elemente; Varianz/Standardabweichung der Population |
| dot(a,b), norm(a) | Vektor-Skalarprodukt, euklidische bzw. Frobenius-Norm |
| matmul(a,b), transpose(a), solve(a,b) | Matrixmultiplikation, Transponierung, quadratisches Gleichungssystem mit Vektor-RHS |
| sin(a), cos(a), exp(a), log(a), sqrt(a) | Elementweise Mathematik |
| get(a,i,j), set(a,i,j,x) | Matrixelement lesen oder ändern |

`+ - * /` sind elementweise mit Broadcasting (beide Operandenreihenfolgen);
Matrixmultiplikation ist ausdrücklich `matmul`. Vektoren unterstützen Indexzugriff und
Indexzuweisung; Matrizen liefern mit `a[i]` eine **Kopie** der Zeile. Iteration über
Matrizen liefert Zeilenkopien. Negative Indizes funktionieren. `len` gibt die erste Dimension an.
Arrays sind referenzsemantisch, arithmetische Ergebnisse neue Arrays, Gleichheit ist
Objektidentität, keine elementweise Maske. Explizite Views, Reshape und Permute sind verfügbar; die allgemeine Slice-Syntax gilt weiterhin nur für Listen und Strings. Keine GPU-Unterstützung.

Maximal 16 Millionen Elemente pro Array (auch das Produkt der Dimensionen mit 0 durch 1 ersetzt muss innerhalb dieser Grenze liegen); diese Grenze ist keine globale Speicherbeschränkung.
Division und mathematische Funktionen folgen IEEE-754 (NaN/Inf möglich). Ganzzahlen
werden bei Array-Konvertierung auf Float64 gerundet. Leere Mittelwert-/Varianz-/Min-/Max-Reduktionen
lösen einen Fehler aus. `solve` verwendet Gauß-Elimination mit partieller Pivotisierung;
keine Konditionszahl- oder Fehlerabschätzung, nicht für schlecht konditionierte Systeme validiert.
Das Projekt ist ein experimenteller Prototyp, keine validierte wissenschaftliche Software.

`./arda examples/science.arda` führt lineare Regression und ein Gleichungssystem aus.
`python3 benchmarks/science_bench.py ./arda` misst Skalarprodukt und Matrixmultiplikation.
Die gemessenen Resultate stehen in `benchmarks/science-results.txt`; Datenaufbau und
Prozessstart werden bei diesen Benchmarks ausgeschlossen. Die BLAS-Option betrifft nur
`matmul`, nicht `dot` oder `solve`.

## 0.10 — Anor: Tensoren

```arda
tol numc as nc
esta a = nc.reshape(nc.linspace(1, 12, 12), [2, 2, 3])
pedo(a + nc.array([10, 20, 30]))
esta v = nc.view(a, 2, 0, 3, 2)
nc.put(v, [1, 0, 1], 99)
pedo(nc.at(a, [1, 0, 2]))
```

| API | Semantik |
|---|---|
| `array(nested_list)`, `zeros(d1, ...)`, `ones(d1, ...)` | Float64, Rang 1–8; keine ragged Arrays |
| `reshape(a, shape)` | Zero-copy, zusammenhängender Input und identische Elementzahl nötig |
| `view(a, axis, start, stop, step)` | Zero-copy, nichtnegative Grenzen, exklusiver Stop, positiver Step |
| `permute(a, axes)` | Zero-copy, vollständige Achsenpermutation |
| `copy(a)` | Unabhängige zusammenhängende Kopie |
| `at(a, indices)`, `put(a, indices, value)` | Einzelwert, negative Indizes zulässig |
| `contiguous(a)`, `simd()` | Layout und kompiliertes SIMD-Backend |

Row-major Strides in Elementen; Broadcast-Achsen nutzen Stride 0. Ein Odometer
läuft über äußere Dimensionen und übergibt die innerste Dimension an einen SIMD-
oder skalaren Kernel. SSE2 (128 Bit, zwei Float64-Werte) für vier elementweise
Operatoren; portabler Fallback. Keine AVX- oder GPU-Implementierung. Views halten
via GC-Referenz den Pufferbesitzer am Leben; Mutationen sind sichtbar. `a[i]`
bei Rang >1 und das bisherige `transpose()` bleiben Kopien. `bytes()` ist die
logische Nutzdatenmenge, nicht exklusiv zugewiesener Speicher. Nichtkontiguierliche
Inputs werden für manche ältere Reduktionen und lineare Algebra materialisiert.
Matrixmultiplikation bleibt auf Rang 2 begrenzt; blockierter C-Kernel (32er-Blöcke)
oder CBLAS dgemm. Kein batched matmul und keine Fusion verketteter Array-Ausdrücke.

## 0.11 — Narya: numerische Verfahren

| API | Algorithmus |
|---|---|
| `calc.trapz(y, h)` | zusammengesetzte Trapezregel, gleichmäßiges Gitter |
| `calc.simpson(y, h)` | zusammengesetzte Simpson-1/3-Regel; ungerade Anzahl >=3 Stichproben |
| `ode.integrate(f, a, b, n=100)` | Simpson-Regel mit Arda-Callback, n gerade Intervalle |
| `ode.rk4(f, t0, y0, h, steps)` | klassisches RK4, Skalar oder Array; Liste inkl. Anfangszustand |
| `calc.rk4_linear(A, y0, h, steps)` | natives RK4 für y'=Ay, Matrix mit einer Zeitstufe pro Zeile |
| `calc.fft(a)`, `calc.ifft(a)` | Radix-2 Cooley–Tukey DIT, Nx2 Real/Imaginär-Format |
| `calc.lapack_solve(A,b)` | optional LAPACK DGESV (LU mit partieller Pivotisierung) |
| `calc.eigh(A)` | optional LAPACK DSYEV, symmetrische reelle Matrix; [Eigenwerte, Eigenvektoren als Spalten] |
| `calc.backend()` | LAPACK oder portable-c |

FFT: Bit-Reversal-Permutation, log2(N) Butterfly-Stufen; Vorwärtstransformation
mit negativer Phase ohne Normierung, inverse mit positiver Phase und Faktor 1/N.
N muss Zweierpotenz bis 1,048,576 sein. Keine beliebigen FFT-Längen oder komplexen
Sprachwerte. Simpson und Trapezregel nutzen gleichmäßig verteilte Stützstellen;
für glatte Integranden sind globale Fehler O(h^4) bzw. O(h^2). RK4 benötigt vier
Ableitungsauswertungen je Schritt, globale Ordnung 4 für hinreichend glatte nichtsteife
Probleme. Keine adaptive Schrittweite, Ereigniserkennung oder implizite Solver.
Das eingebettete `ode`-Modul (`lib/ode.arda`) führt Callbacks über die reguläre VM
mit normalem GC und Exception-Handling aus; Callbacks sollen Eingaben nicht verändern.
LAPACK nutzt die übliche LP64 Fortran-Schnittstelle der Linux-Bibliothek; Layout-
Konvertierung row-major -> column-major; `eigh` fordert exakt symmetrische Eingaben.

```sh
sudo apt install libblas-dev liblapack-dev
make BLAS=1 LAPACK=1
make test BLAS=1 LAPACK=1
```

Ohne LAPACK lösen die beiden LAPACK-Funktionen einen verständlichen Fehler aus;
`nc.solve` bleibt im portablen Build verfügbar. Seit 0.13.1 ist `make clean` beim Wechseln der Build-Optionen nicht mehr nötig.

## 0.12 — Nenya: Reverse-Mode Autodiff

```arda
tol ad
esta x = ad.variable(3)
esta y = ad.variable(2)
esta f = x*x + x*y
ad.backward(f)
pedo(ad.value(f), ad.grad(x), ad.grad(y))
```

Ausgabe: `15 8 3`. Jeder skalare Graphknoten enthält Wert, Gradient, maximal zwei
Eltern und lokale partielle Ableitungen. Die Vorwärtsauswertung baut einen DAG.
Eine iterative Tiefensuche erstellt die Postorder, dann wird in umgekehrter
Reihenfolge die Kettenregel angewandt; Gradienten gemeinsamer Teilgraphen werden
addiert. Zeit und Speicher O(V+E). `backward` setzt die Gradienten der vom Ziel
erreichbaren Knoten zurück; nicht erreichbare Knoten bleiben unverändert.
Graphreferenzen werden vom GC verfolgt, Freigabe wenn der Graph unerreichbar ist.
`ad.grad` vor backward ist 0, keine Warnung. Neue Graphen für neue Parameterwerte.

| API | Funktion |
|---|---|
| `variable(x)`, `value(x)`, `grad(x)`, `backward(loss)` | Graph und Gradienten |
| `+ - * /`, unäres `-` | differenzierbare skalare Operatoren |
| `sin`, `cos`, `exp`, `log`, `sqrt`, `tanh`, `pow(x, exponent)` | lokale analytische Ableitungen; konstanter numerischer Exponent |
| `sgd(params, gradients, lr)` | neue Float64-Parameter als Vektor |
| `adam(params, gradients, m, v, step, lr)` | [neue Parameter, neuer erster Moment, neuer zweiter Moment] |

SGD: p <- p - lr*g. Adam: beta1=.9, beta2=.999, epsilon=1e-8; exponentielle
Momente, Bias-Korrektur mit 1-beta^t, t beginnt bei 1. Updates liefern neue Arrays.
`examples/learning.arda` trainiert eine lineare Regression aus vier Stichproben
mit skalarem AD und Adam. Noch keine automatische Tensor-Differentiation,
höheren Ableitungen, neuronalen Layer-APIs, GPU oder Minibatch-Datenpipeline.
Vergleiche von AD-Knoten erfordern `ad.value`. Gradienten außerhalb der definierten
reellen Domäne bzw. nichtendliche Werte führen zu Fehlern. Nicht als validierte
wissenschaftliche Software verwenden.

## Neue Beispiele und Nachweise

```sh
./arda examples/tensors.arda
./arda examples/calculus.arda
./arda examples/learning.arda
make test
make sanitize
```

Siehe `benchmarks/v012-validation.txt` für tatsächliche Test- und Beispielausgaben.
Ältere Benchmark-Dateien dokumentieren historische Versionen, keine neuen Messungen.

## Sprache

Eine Anweisung pro logischer Zeile. Blöcke beginnen mit `:` und einem eingerückten Block; vier Leerzeichen sind empfohlen. Semikolons und Tabs zur Einrückung werden abgelehnt. Geschweifte Klammern stehen ausschließlich für Dictionaries. In `()`, `[]` und `{}` sind Zeilenumbrüche erlaubt. Kommentare beginnen mit `#` oder `//`. Einfache und doppelte Anführungszeichen sind möglich.

```arda
esta hero = "Frodo"
esta items = [1, 2, 3]
esta powers = {"Nenya": 10, "Vilya": 20}

caro square(x):
    nan x * x

an x in items:
    pedo(square(x))

qui powers["Nenya"] > 5:
    pedo(hero)
ai:
    pass
```

| Arda | Alternative | Bedeutung |
|---|---|---|
| esta | let | neue veränderbare Variable |
| caro | fn, def | Funktion |
| nan | return | Rückgabe |
| qui | if | Bedingung |
| ai | else | Alternative |
| tenna | while | While-Schleife |
| an | for | Iteration |
| pedo | print | Ausgabeanweisung |
| nae | true | wahr |
| law | false | falsch |
| mun | nil | Nullwert |
| nost | class | Klasse |
| tol | import | Modulimport |

`elif`, `in`, `as`, `break`, `continue`, `and`, `or`, `not`/`!`, `try`, `catch`, `throw`/`raise`, `pass` sind ebenfalls verfügbar. Die englischen Formen und die Arda-Formen können gemischt werden.

### Werte und Scopes

- Zahlen sind entweder vorzeichenbehaftete 64-Bit-Integer (`int`) oder IEEE-754 `double` (`float`). Integer-Überläufe lösen Runtime-Fehler aus. `/` liefert einen Float, `%` benutzt C-Restsemantik. Gemischte Arithmetik liefert Float. Strings sind gültiges UTF-8; Länge, Index und Iteration zählen Unicode-Codepoints, nicht Graphemcluster. Bool, nil, Listen, Dictionaries, Funktionen, native Funktionen, Klassen, Instanzen, gebundene Methoden und Module sind dynamische Werte.
- `esta` deklariert eine Variable im aktuellen lexikalischen Scope. `x = ...` aktualisiert die nächste vorhandene Deklaration; die Built-ins sind nicht direkt überschreibbar, können aber mit `esta` lokal verdeckt werden. Blöcke und einzelne for-Iterationen besitzen eigene Scopes.
- Funktionen sind First-Class Values, unterstützen Rekursion und Closures. Optionale Standardparameter werden bei jedem Aufruf von links nach rechts ausgewertet (anders als Python). Sie dürfen frühere Parameter referenzieren. Keine Keyword-Argumente oder variadischen Parameter.
- Operatoren: `+ - * / %`, `== != < <= > >=`, `and or`, `not !`, unäres `-`. `and/or` sind kurzschließend und geben einen Operanden zurück. `not` hat dieselbe hohe Präzedenz wie `!`; Klammern verwenden, wenn ein ganzer Vergleich negiert werden soll. Vergleiche werden nicht wie in Python verkettet. `+` mit einem String konvertiert den anderen Operanden in einen String. Listen/Dictionaries werden bei Gleichheit nach Identität verglichen.
- Listen und Strings unterstützen negative ganzzahlige Indizes. Listen können per Index verändert werden. Dictionary-Schlüssel sind Strings, Iteration liefert Schlüssel in nicht garantierter Reihenfolge (Snapshot der Schlüssel beim Beginn der Iteration). Listeniteration läuft über die aktuelle Liste. Fehlende Schlüssel und ungültige Indizes lösen Runtime-Fehler aus.

### Klassen (0.6)

```arda
nost Ring:
    caro init(self, name, power):
        self.name = name
        self.power = power

    caro charge(self, amount):
        self.power = self.power + amount
        nan self.power

esta ring = Ring("Nenya", 10)
esta charge = ring.charge
pedo(charge(5))
```

`init` oder `__init__` dient als Konstruktor, dessen Rückgabewert ignoriert wird. Methoden benötigen explizit `self` als ersten Parameter, beim Aufruf einer gebundenen Methode wird dieser automatisch übergeben. Instanzattribute entstehen bei Zuweisung. Klassenblöcke erlauben Methoden und `pass`. Einfache Vererbung mit `nost Child(Parent):`, einschließlich geerbter Konstruktoren und überschreibbarer Methoden. Basismethoden können explizit als `Parent.method(self, ...)` aufgerufen werden. Kein `super`, keine Mehrfachvererbung, Properties oder Operator-Overloading.

### Module (0.6)

Datei `forge.arda`:

```arda
esta metal = "mithril"
caro double(x):
    nan x * 2
```

Datei `main.arda` im selben Verzeichnis:

```arda
tol forge
import "./forge.arda" as f
pedo(forge.metal, f.double(5))
```

Imports werden relativ zur importierenden Datei aufgelöst (`.arda` wird bei Bedarf ergänzt), der REPL-Import relativ zum Arbeitsverzeichnis. Module besitzen eigene globale Umgebungen und teilen die Built-ins. Externe Zugriffe auf Namen, die mit `_` beginnen, werden verweigert. Die Module werden pro VM anhand des kanonischen Pfads gecacht und einmal ausgeführt. Zirkuläre Imports werden abgelehnt. Nach einem Runtime-Fehler beim Laden wird das Modul als fehlgeschlagen markiert. Wenn keine passende Datei existiert, können die eingebauten Module math, time, text, io und collections importiert werden. Kein Paketmanager oder natives dynamisches Laden.

### Fehlerbehandlung (0.6)

```arda
caro fail():
    throw "The ring is lost"

try:
    fail()
catch error:
    pedo(error)
```

`catch` bindet eine Fehlermeldung als String. Auch Runtime-Fehler wie Division durch Null, falsche Argumentanzahl und fehlende Namen werden abgefangen. Unbehandelte Fehler zeigen Dateipfad, Zeile und Call Stack und beenden den Dateilauf mit Status 1. Lexer-, Parser- und Compilerfehler sind nicht mit `catch` abfangbar. Kein `finally` oder typisierte Exceptions.

### Standardfunktionen

| Funktion | Bedeutung |
|---|---|
| len(x) | Länge von Liste, String (Codepoints), Dictionary |
| push(list, x), pop(list) | Liste verändern |
| str(x), num(x) | Konvertierung |
| sqrt(x) | Quadratwurzel |
| range(end), range(start,end[,step]) | Liste ganzzahliger Werte, höchstens 1 Mio. Elemente |
| keys(map) | Liste der Dictionary-Schlüssel |
| type(x) | dynamischer Typ als String |
| read_text(path) | Textdatei lesen (keine NUL-Bytes) |
| write_text(path, text) | Textdatei überschreiben, Anzahl geschriebener Bytes |
| heap(), heap_bytes() | verwaltete Objekte bzw. deren Speicher |
| clock_gc(), collect() | abgeschlossene GC-Zyklen bzw. laufenden/neuen Zyklus abschließen |

Dateioperationen verwenden das Arbeitsverzeichnis des Prozesses (anders als Modulimporte). Nur vertrauenswürdige Programme ausführen: Arda besitzt keine Sandbox und kann mit den Rechten des Prozesses Dateien lesen und schreiben.

## Architektur und Entwicklungsphasen

### 0.4: Stack-VM

`src/lexer.c` -> `src/parser.c` -> `src/compiler.c` -> `src/vm.c`.
Der Lexer generiert NEWLINE, INDENT und DEDENT. Der rekursive Parser erzeugt einen AST. Der Compiler erzeugt pro Funktion eine Bytecode-Sequenz aus 56 Opcodes mit Operand, Zeilennummer und Konstantenpool. Nach dem Kompilieren werden AST und Tokens freigegeben. Die VM verwendet einen Operandenstack, Call Frames, lexikalische Environments mit Hash-Tabellen sowie Sprungbefehle. Sie benutzt keine rekursiven C-Aufrufe für Arda-Funktionsaufrufe. Bytecode ist ein internes In-Memory-Format, kein stabiles serialisiertes Dateiformat.

### 0.5: Mithril GC

`src/gc.c` implementiert inkrementelles Tracing mit nichtrekursivem Gray-Stack, Write Barriers (Insertion Barrier), Root-Rescans und atomarem Sweep. Der Collector startet bei einem dynamischen Schwellwert (anfänglich 64 KiB verwalteter Objektspeicher) und arbeitet zwischen VM-Instruktionen. Normale Mark-Schritte behandeln bis zu 32 Objekte; große Objekte und Root-Scans können dennoch mehr Arbeit verursachen. Neue Objekte werden während eines laufenden Zyklus grau markiert. Stack, Call Frames, Exception-Handler, globale Scopes und Modulcache zählen als Wurzeln. Zyklen sind sammelbar. `--gc-stress` startet bzw. bearbeitet die Collection an jeder Instruktionsgrenze. `collect()` beendet einen GC-Zyklus sofort; Floating Garbage kann bis zum folgenden Zyklus erhalten bleiben.

`heap_bytes()` umfasst Objekte und deren dynamischen Speicher (Strings, Listenpuffer, Hash-Tabellen, Schlüssel), nicht VM-Stack, Gray-Stack, Compiler-Pools, Modulcache-Tabelle, Interning-Buckets oder allgemeine Allocator-Kosten. Kompilierte Programme bleiben bis zum Ende der VM erhalten, damit Closures auf gültigen Bytecode zeigen. Module bleiben im Cache. GC ist nicht generational, Sweep und einzelne große Trace-Schritte sind nicht inkrementell. Die Runtime garantiert keine Echtzeit-Pausengrenze.

### 0.6: Objektmodell, Module und Fehler

`src/vm.c` verwaltet Klassen, Instanzen, gebundene Methoden, Modulcache sowie Handler-Stack und Exception-Unwinding. `src/native.c` liefert native Standardfunktionen. `src/main.c` enthält CLI und REPL. `src/arda.h` enthält gemeinsame Datentypen und interne APIs. `src/util.c` enthält gemeinsame Speicher-/Dateihilfen.

## Werkzeuge

```sh
./arda --tokens examples/fibonacci.arda
./arda --bytecode examples/fibonacci.arda
./arda --gc-stress examples/rings.arda
./arda examples/memory.arda
./arda
```

REPL: eine einfache Anweisung pro Zeile; eingerückte Blöcke mit einer leeren Zeile abschließen; `:quit` beendet die Sitzung. Runtime-Fehler erlauben weitere Eingaben, Syntaxfehler beenden den Prozess. Mehrzeilige Literale außerhalb eines Blocks sind in der REPL noch nicht komfortabel unterstützt.

## Neu in 0.7: Shadowfax

- `Val` verwendet eine getaggte Union statt separater Zahl- und Objektfelder. Kein NaN-Boxing.
- Funktionsparameter liegen in indizierten Slots der Funktionsumgebung; `GET_LOCAL`/`SET_LOCAL` greifen direkt darauf zu. Andere lokale Namen und eingefangene Variablen nutzen weiter lexikalische Namensauflösung. Closures behalten ihre Umgebung. Keine vollständige Stack-Slot-/Upvalue-Architektur.
- Leere lexikalische Scopes werden beim Kompilieren vermieden, wenn ein Block keine eigenen Bindungen enthält. `if` ohne `else` benötigt keinen zusätzlichen unbedingten Sprung.
- Ein schwacher String-Interning-Hashindex teilt identische Strings, ohne unerreichbare Strings dauerhaft festzuhalten. Der GC entfernt tote Interning-Einträge vor Sweep.

## Neu in 0.8: Rivendell

```arda
nost Hero:
    caro init(self, name, level = 1):
        self.name = name
        self.level = level

nost Elf(Hero):
    pass

esta hero = Elf("Galadriel")
pedo(hero.name, isinstance(hero, Hero))
esta square = lambda x: x * x
pedo([square(x) an x in range(6) qui x % 2 == 0])
pedo("Mithríl 🌿"[-1], [1, 2, 3][1:])
tol math
pedo(math.sqrt(81))
```

- List Comprehensions mit einem `an`/`for` und optionalem `qui`/`if`-Filter; verschachtelte Comprehensions als eigene Ausdrücke sind möglich. Jede Iteration besitzt einen eigenen Scope.
- Lambdas mit positionalen Parametern und einem Ausdruck. `lambda: expr` ist möglich.
- Slicing `[start:end]` mit optionalen Grenzen, negativen Indizes, ohne Schrittweite. Liefert neue Liste oder String.
- UTF-8 wird in Quelltext und Textdateien geprüft; Unicode-Identifier akzeptieren gültige Nicht-ASCII-Codepoints, aber keine Unicode-XID-Klassifikation oder Normalisierung. Keine Graphemsegmentierung oder Unicode-Casefolding.
- Neue Built-ins: `int`, `float`, `abs`, `sin`, `cos`, `floor`, `ceil`, `monotonic`, `split`, `join`, `startswith`, `endswith`, `isinstance`. `type(1)` liefert `int`, `type(1.5)` liefert `float`. `num` erkennt auch Ganzzahlstrings. Die bisherige `number`-Typbezeichnung wurde ersetzt.

| Eingebautes Modul | Funktionen |
|---|---|
| math | sqrt, abs, sin, cos, floor, ceil, pi |
| time | monotonic |
| text | split, join, startswith, endswith |
| collections | len, push, pop, range, keys |
| io | read_text, write_text |

## Tests und Benchmarks

69 Tests (62 bisherige Tests + 7 neue Testmethoden) bestanden im optimierten C11-Build und mit ASan/UBSan. `examples/quest.arda` wurde zusätzlich mit aktivem LeakSanitizer und GC-Stress erfolgreich ausgeführt. Syntaxfehler verlassen weiterhin den Prozess, daher ist LeakSanitizer für die gesamte Suite nicht aktiviert.

```sh
make test
make sanitize
make
./arda examples/hello.arda
./arda examples/eldar.arda
/usr/bin/python3 benchmarks/run.py /path/to/arda-0.8.0 --rounds 9
```

`benchmarks/results.txt` enthält den gemessenen Median aus neun abwechselnden Läufen je Runtime und Workload einschließlich Prozessstart, Einzelwerte und geprüfte Ausgaben. Beide Arda-Versionen wurden mit identischen GCC-14.2-O2-Optionen kompiliert. Vergleich mit unverändertem Arda 0.8.0 und System-CPython 3.13.5, CPU-Affinität 0. Kein umfassender Performance-Nachweis. Für Wiederholbarkeit kann eine verfügbare CPU mit `--cpu N` festgelegt werden.

## Offene Punkte

Vollständige lokale Stack-Slots, NaN-Boxing, umfassende Bytecode-Optimierung, Memory Pools, inkrementelles Sweeping, GC-verwaltete Bytecode-Prototypen, beliebig große Integer, vollständige Unicode-Semantik, f-Strings, Sets/Tupel, Generatoren, step-Slicing, Keyword-Argumente, Typannotationen, JSON/Netzwerk/OS-Module, FFI, Paketmanager, Debugger und LSP sind noch nicht implementiert. Range erzeugt weiterhin eine Liste mit maximal einer Million Elementen und beschränkten ganzzahligen Grenzen. Kein JIT. Die maximale Funktionsaufruftiefe beträgt 512. Syntaxfehler-Recovery und Ressourcenlimits bleiben ausbaufähig.

MIT-Lizenz, siehe LICENSE. Unabhängiges Fan-Projekt, keine Verbindung zum Tolkien-Nachlass.

## Neu in 0.8.1: Performance

- Compilerseitig vorberechnete Namens-Hashes und VM-lokale Tabellenlayout-Stamps: ein konstanter Name kann einen Tabellenindex cachen. Stamps ändern sich beim Resize; der Cache enthält keine GC-Pointer. Prototypen gehören jeweils genau einer VM.
- Sechs kombinierte Opcodes: `SET_DROP`, `SET_LOCAL_DROP`, `ADD_INT_CONST`, `SUB_INT_CONST`, `MUL_INT_CONST`, `JFALSE_LT_CONST`. Dynamische Fallbacks behalten bisheriges Verhalten bei; Integer-Overflow bleibt ein Laufzeitfehler.
- Funktionsumgebung und Parameterslots werden zusammen alloziert. Inline-Value-Helfer, schneller GC-Schwellwerttest und weniger Stack-Operationen.
- Keine Änderung an der Sprachsyntax. Kein JIT, keine vollständig direkten lokalen Slots, keine öffentliche Embedding-API.

Praktisches Beispiel: `./arda examples/quest.arda` simuliert eine Quest-Belohnung in einer Spielregel-Engine. Die C-VM ist nicht sandboxed; untrusted Skripte nicht direkt ausführen.

Gemessene Mediane (ms):

| Workload | 0.8.0 | 0.8.1 | CPython |
|---|---:|---:|---:|
| fib(25) | 49.476 | 29.642 | 25.639 |
| 1 Mio. Schleifendurchläufe | 192.262 | 94.800 | 130.638 |
| 100.000 Methodenaufrufe | 41.875 | 28.164 | 28.415 |
| 100.000 Listenallokationen | 61.611 | 42.778 | 37.023 |

## Arda 0.13.0 — High-dimensional scientific computing

### Build and run

```sh
make
make test
./arda examples/highdim.arda
make sanitize
# Optional external linear-algebra backend:
make BLAS=1 LAPACK=1
make test BLAS=1 LAPACK=1
```

The runtime is C11; Python is only needed to run the tests, regenerate embedded standard-library modules and run the benchmark driver. No Python is needed for Arda programs. The previous eight-axis / 16,000,000-element per-array limit remains in place. Multiple live arrays may consume considerably more RAM. No GPU or multithreading is implemented in this release.

### Module `high` (native C)

| Function | Contract |
|---|---|
| `cg(A,b,tolerance,maxiter)` | Jacobi-preconditioned CG for a nonempty symmetric positive-definite square matrix and vector. Starts at zero. Matrix symmetry is checked exactly; positive definiteness is a caller assumption (nonpositive curvature is detected when encountered). |
| `poisson(b,tolerance,maxiter)` | Matrix-free CG for the negative discrete Laplacian on a rectangular Cartesian tensor grid (rank 1–8), homogeneous Dirichlet conditions outside the grid and unit spacing. Starts at zero. For equal physical spacing h, pass h*h times the physical RHS. |
| `laplace(x)` | Applies that same **negative** discrete Laplacian, returns matching shape. |
| `lstsq(A,b,tolerance,maxiter)` | CGLS for rectangular dense matrices, starting at zero. Does not form AᵀA. For ill-conditioned systems or rank-deficient systems use a validated QR/SVD implementation where possible. |
| `axpy(x,y,a)` | Returns `a*x+y` in one pass, no intermediate scaled tensor; exact shape match. Not necessarily a hardware fused multiply-add. |
| `inner(x,y)` | Full elementwise inner product over matching tensors of any rank, extended-precision partial accumulators where supported by the C compiler. |
| `sum_axis(x,axis)` | Sum over one axis (negative indices allowed), drops that axis; rank-one input produces a scalar. Empty reduction sums to zero. |
| `finite(x)` | Whether a numeric scalar or all elements of a tensor are finite. |
| `rkerror(y,z,low,rtol,atol)` | Weighted RMS error estimator for RK45 (scalar or matching nonempty tensors). |

The three iterative linear solvers return `[solution, iterations, relative_residual, converged]`. `cg` and `poisson` use `norm(b-Ax)/norm(b)`, `lstsq` uses `norm(Aᵀ(b-Ax))/norm(Aᵀb)` (stationarity, not necessarily a small data residual). Zero RHS/initial normal residual yields a zero solution and zero iterations. The tolerance must be finite and in `(0,1)`. `maxiter` is an integer in `1..1000000`. Reaching the limit returns `law` if the recomputed residual does not meet tolerance; numerical breakdown throws. Finite precision, scaling and conditioning limit attainable accuracy. Dense CG per-iteration cost O(n²); CGLS O(mn); Poisson O(dN) for d axes and N grid points, with O(N) solver work space. The native CG work consists of five full arrays including solution (plus RHS and any inputs/copies); native calls do not yield or run GC until they return.

Poisson uses a 2d+1 point stencil: diagonal `2*d`, subtract existing immediate neighbors in each dimension. For row-major strided input, contiguous copies are used when necessary. It does not form an N×N matrix. No arbitrary sparse matrix API, anisotropic spacing, mixed boundary conditions, multigrid or general PDE discretization in this version.

### Module `ode`: adaptive Dormand–Prince 5(4)

`ode.rk45(f,t0,y0,t1,h,rtol=1e-6,atol=1e-9,maxiter=100000)` returns `[final_state,accepted_steps,rejected_steps]`. `f(t,y)` is a user-defined derivative, y is a number or Float64 tensor; callers must provide shape-compatible derivatives and avoid mutating state. Positive initial step magnitude, forward/backward integration. Seven stage evaluations per attempted step, embedded 5th/4th order estimates; normalized RMS local error uses `atol+rtol*max(abs(y),abs(y_new))` per component. Accept at error <= 1, adapt step by `clamp(0.9*err^(-1/5),0.2,5)` (5 when error is zero). A native C error kernel avoids temporaries and uses extended precision for scaling. Final step clipped to target time. Throws on step-limit/underflow or nonfinite state; no dense output/events, FSAL reuse, or implicit stiff solver. Error tolerance controls estimated local error, not guaranteed global error. Embedded standard library is implemented in Arda and uses native C tensor kernels.

### Module `opt`: L-BFGS

`opt.lbfgs(f,x0,tolerance=1e-6,maxiter=500,memory=10)` minimizes a differentiable real scalar objective. `f(x)` must return `[value,gradient_tensor]` matching x, without mutating inputs or retaining mutable work buffers. Returns `[x,iterations,gradient_norm,converged,value]`. Limited-memory BFGS stores up to 50 `(s,y)` curvature pairs, uses two-loop recursion and inverse-Hessian scale `(sᵀy)/(yᵀy)`, Armijo backtracking with halved step size (at most 40 trials), skips insufficiently positive curvature pairs, falls back to steepest descent if the direction is not descending. Stop when L2 gradient norm <= tolerance; returns false if budget or line search is exhausted. O(memory*N) state/work per iteration plus callback costs. No constraints, strong-Wolfe search or global convergence guarantee for arbitrary nonconvex problems. The callback API is not tensor autodiff; users supply the gradient. Written in Arda with native array operations. `tools/embed_opt.py` regenerates the header if the source changes.

### Optimization round

- `numc.dot`: four independent long-double accumulators, then combine, reducing serial dependencies while retaining extended precision on x86-64. Changed accumulation order can change rounding; no fast-math flags.
- `numc.matmul`: runtime AVX2 dispatch on GCC-compatible x86 compilers (64×64 cache blocks, four Float64 lanes per vector, unrolled contiguous update); SSE2 fallback (32×32 blocks, two lanes) or scalar fallback on other hardware. External CBLAS takes precedence in BLAS builds. No AVX-512, FMA-specific kernel or GPU.
- `numc.copy` for strided tensors: mixed-radix odometer maintains offsets by increments/rollovers rather than integer division/modulo at every element.
- `high.axpy` fuses scale/add at array-loop level and avoids an extra array allocation. This is not a claim of a single hardware FMA instruction.

Benchmark: both versions built with same default GCC 14.2 `-O2` flags, no external BLAS, same CPU affinity, nine alternating runs, medians of in-program operation timings (input creation/process launch excluded; output allocation included). Verified the output checks for every run. See `benchmarks/highdim-results.txt` for raw timings and `benchmarks/highdim_bench.py OLD NEW` to reproduce. Single host, synthetic inputs, not a broad comparison against optimized NumPy/Julia.

| Workload | 0.12.0 ms | 0.13.0 ms | Speedup |
|---|---:|---:|---:|
| Dot, 2M values | 4.822 | 1.530 | 3.15x |
| 512×512 matmul | 65.018 | 31.551 | 2.06x |
| Strided copy, 2M values | 17.840 | 11.778 | 1.51x |

`examples/highdim.arda` includes 3D Poisson (32,768 unknowns), 4D manufactured Poisson (4,096), 1,000 simultaneous decay ODEs, L-BFGS weighted quadratic (10,000 variables), and 8D manufactured Poisson (65,536). The latter is a symmetric test problem, not a prediction of general iteration counts. See `benchmarks/highdim-example.txt` for actual results. 120 test methods passed (107 existing + 13 new), normal, ASan/UBSan and BLAS/LAPACK builds. Example ran with GC stress and ASan leak detection enabled. These are regression and numerical checks, not independent scientific certification.
