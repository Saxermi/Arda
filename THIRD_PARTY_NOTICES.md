# Third-party notices

Arda's own source and documentation are distributed under the [MIT License](LICENSE). This inventory covers the 0.14.3 source tree, build options and local scientific validation. The repository's native scientific sources and generated Arda library headers were inspected for reuse declarations. No vendored third-party numerical implementation was identified in that inventory. Historical algorithm provenance is tracked in the [research directory](research/README.md).

## Runtime and optional numerical providers

| Component | Use in Arda | License and distribution |
| --- | --- | --- |
| Platform C and math libraries | The default build uses C11 and links `-lm` for elementary functions and scalar math. | Provider-specific. Check the concrete platform runtime when redistributing it. |
| CBLAS provider | Optional `BLAS=1` links system `-lblas` and calls `cblas_dgemm`. OpenBLAS was used in local evaluation. | The OpenBLAS top-level license is BSD 3-Clause. Preserve the applicable upstream and component notices if distributing its code or binaries. Other providers have their own terms. [OpenBLAS 0.3.29 license](https://github.com/OpenMathLib/OpenBLAS/blob/v0.3.29/LICENSE) |
| LAPACK provider | Optional `LAPACK=1` links system `-llapack -lblas` and calls `dgesv`, `dsyev`, `dgeev`, and `dgesvd`. | Reference LAPACK uses BSD-style redistribution terms, including its attribution and disclaimer. Check the actual implementation and ABI supplied by the packaging environment. [Reference LAPACK 3.12.1 license](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/LICENSE) |

The Makefile links libraries installed in the build environment. For a package that bundles libraries, verify the exact binary, upstream revision, transitive components and applicable notices. Keep the notice files required by each component with that distribution. [Build configuration](Makefile)

## Local development and comparison tools

| Component | Use | License source |
| --- | --- | --- |
| Python 3 and its standard library | Build-time header generation, local tests and benchmark harnesses. | Python Software Foundation and historical notices. [Python license](https://docs.python.org/3/license.html) |
| NumPy | Optional reference validation and benchmarks. Listed in `requirements-dev.txt`. | BSD 3-Clause for the NumPy project. Built packages can carry additional bundled notices. [NumPy 2.3.5 license](https://github.com/numpy/numpy/blob/v2.3.5/LICENSE.txt) |
| SciPy | Optional reference validation, Matrix Market comparison and solver benchmarks. Listed in `requirements-dev.txt`. | BSD 3-Clause for the SciPy project. Built packages can carry additional bundled notices. [SciPy 1.17.0 license](https://github.com/scipy/scipy/blob/v1.17.0/LICENSE.txt) |
| Eigen | External C++ baseline discussed for the 0.18 dense benchmark. It is not linked or required by Arda. | Eigen 3.4 is primarily MPL 2.0, with component terms described by the project. [Eigen licensing](https://gitlab.com/libeigen/eigen/-/blob/3.4.0/COPYING.README) |

## Formats, references and project identity

Arda implements Matrix Market exchange against the NIST format description. The bibliography records scientific attribution and contains bibliographic metadata and links. Papers, reference implementations, standard text, model weights and input datasets have their own terms. For any future reuse, record the source revision, license, changes and notice requirements here. [Matrix Market](https://math.nist.gov/MatrixMarket/formats.html) · [Bibliography](research/REFERENCES.bib)

Literary inspiration is described in the README. Use original logos, artwork and documentation. Permission to use Arda under MIT applies to Arda's licensed material. [Project attribution](README.md#license-and-attribution)

## Preliminary patent research

A targeted reading for planned quantization identified two application publications whose claims merit review if Arda adopts similar designs. Claim 1 of US20230350678A1 describes a single processor instruction that quantizes elements from input vector registers and packs them into an output register. Claim 1 of WO2025165444A1 describes a combination of codebook rank reduction, optimization against layer reconstruction error and quantization of a resulting factor. These specific combinations are design-review leads for future fused instructions or learned codebook compression. The roadmap has no concrete implementation of either to compare claim by claim. Family status, territorial coverage and applicability remain unverified. [US application](https://patents.google.com/patent/US20230350678A1/en#claims) · [PCT publication](https://patents.google.com/patent/WO2025165444A1/en#claims)

Before integrating a concrete novel mechanism, search by its claim elements, check the relevant family and current status, and escalate credible unresolved overlap. A preliminary search cannot establish freedom to operate. [WIPO PATENTSCOPE](https://www.wipo.int/en/web/patentscope) · [EPO Espacenet](https://www.epo.org/en/searching-for-patents/technical/espacenet) · [WIPO patent FAQs](https://www.wipo.int/en/web/patents/faq_patents)
