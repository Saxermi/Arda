# Scientific types

## SI quantities

The `units` module stores an immutable Float64 value in coherent SI units with a display unit and seven integer dimension exponents. A quantity can hold a real scalar or a Float64 tensor. Construction snapshots tensor inputs, including strided views. Extracting a tensor value returns a new Float64 array. Conversion to another display unit shares the immutable SI value.

| API | Result |
| --- | --- |
| `quantity(value,unit)` | Interpret a scalar or tensor in the supplied unit |
| `value(q[,unit])` | Return a scalar or independent array in the display unit or specified compatible unit |
| `convert(q,unit)` | Return a quantity with the requested display unit |
| `unit(q)` | Return the display unit's spelling |
| `dimension(q)` | Return exponents in the order length, mass, time, electric current, temperature, amount of substance, luminous intensity |
| `matmul(a,b)` | Multiply two rank-2 quantity tensors using the regular Float64 matrix kernel |
| `sum(q[,axis])` | Sum every element or reduce a single axis with the same unit |

The infix operations `+`, `-`, `*`, and `/` support compatible quantities and scalar or legacy Float64 array operands. Plain operands are dimensionless. Addition and subtraction require matching dimensions. Multiplication and division combine exponents and use a coherent SI display unit. Tensor arithmetic follows the existing Float64 broadcasting rules. Dimension checking occurs before calling the kernel. Plain tensor operations keep their existing layout and kernel dispatch.

The fixed symbol table accepts `1`, `%`, `rad`, `m`, `mm`, `cm`, `km`, `g`, `kg`, `mg`, `s`, `ms`, `us`, `ns`, `min`, `h`, `A`, `K`, `degC`, `degF`, `deltaK`, `deltaC`, `deltaF`, `mol`, `cd`, `Hz`, `N`, `Pa`, `J`, `W`, `V`, and `L`. Compose them using `*`, `/`, and signed integer powers such as `kg*m/s^2`. Operators are applied left to right after powers. Exponents and resulting dimensions must stay within ±32. Input and output values use finite Float64 arithmetic. [SI Brochure, ninth edition](https://www.bipm.org/documents/20126/41483022/SI-Brochure-9-EN.pdf)

### Temperature

Standalone `K`, `degC`, and `degF` are absolute temperatures. Use `deltaK`, `deltaC`, and `deltaF` for temperature differences. The compound grammar also accepts `K` as a linear temperature factor, as in `J/K`. Offset symbols `degC` and `degF` are valid only as standalone absolute temperatures.

Absolute temperatures must be at least zero kelvin. Subtracting two absolute temperatures gives a difference in `deltaK`. Adding or subtracting a difference from an absolute temperature preserves the absolute temperature kind. The runtime rejects adding two absolute temperatures, subtracting an absolute temperature from a difference, multiplying, dividing or summing absolute temperatures, and converting between temperature kinds. [NIST temperature conversions](https://www.nist.gov/pml/owm/si-units-temperature)

## Shape and dtype contracts

`tensor.contract(x,shape[,dtype])` returns the original tensor after checking its rank, its specified dimensions and any specified dtype. A `mun` dimension matches any extent, and an omitted or `mun` dtype matches any dtype. Checks inspect metadata only and support legacy Float64 arrays, typed tensors, mapped tensors, and the Float64 low-rank formats.

`tensor.spec(shape,dtype)` describes a logical tensor without allocating its payload. The returned map contains `shape`, `dtype`, `elements`, `bytes`, and `materializable`. Each extent is a nonnegative signed 64-bit integer. Rank is 1 to 8. A count beyond signed Int64 is reported as `mun`. A zero extent makes the logical element and byte counts zero. `materializable` uses the heap tensor layout cap, including the stride span for empty shapes. Actual allocation also depends on the active memory budget. Live tensor layouts retain their compact per-axis indices.

## Static notation and validation

`getTex` displays bracket index expressions using mathematical subscripts. Its assumptions report unresolved bounds and shapes. The query is static and can document a function whose inputs are unavailable. Runtime shape and dtype checking is explicit through `tensor.contract`. [MathIR rules](mathdoc.md#subscripts)

The independent NumPy suite checks conversion constants, temperature offsets, tensor broadcasting, signed-stride snapshots, matrix products, and reductions. [Runnable example](../../examples/scientific_types.arda) · [Numerical suite](../../tests/reference_mithril_numerical.py)
