# Symbolic PR-EOS: Code-Only Release

Notebook-based implementation accompanying **Symbolic Machine Learning for
Vapor-Liquid Equilibrium Prediction in Nitrogen + n-Alkane Mixtures**.

This release demonstrates symbolic discovery, joint training of shared basis
functions and carbon-number coefficients, and temperature-panel VLE plotting
for **C5, C6, C7, C9, C10, and C12 + N2**.

**No experimental data, property datasets, prediction CSVs, trained checkpoints,
or scientific example figures are included.** Obtain the input data separately
and prepare the formats in [data/README.md](data/README.md). An optional,
explicitly fictional demonstration checks the software without external data.

## Model Comparisons

The plotting code supports experimental observations and four model traces:

1. PR-EOS with `kij = 0`.
2. PR-EOS with a system-specific constant calibrated BIP.
3. PR-EOS with the generalized BIP correlation of Yang et al. (2022).
4. Two-level symbolic regression (proposed).

Yang's implementation uses **PR78** and its published component-property
correlations. The figure label is shortened to `PR-EOS (generalized kij)`.
The zero/calibrated baselines use PR76. This compares complete literature
models rather than isolating BIP alone with all other EOS settings identical.

No validation-system workflow, combined-isotherm plot, critical-locus figure,
review correspondence, or manuscript PDF is included.

## Installation

Use Python 3.10 or 3.11. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name symbolic-pr-eos --display-name "Python (Symbolic PR-EOS)"
jupyter lab
```

On Windows, activate with `.venv\Scripts\activate` in Command Prompt, or
`.\.venv\Scripts\Activate.ps1` in PowerShell. Choose the installed kernel in
Jupyter. Training uses CPU; a GPU is not required.

For optional symbolic discovery, also install:

```bash
python -m pip install -r requirements-discovery.txt
```

PySR uses Julia through `juliacall`. The first import may download Julia and
packages and requires network access and writable package directories. See the
[PySR installation guide](https://ai.damtp.cam.ac.uk/pysr/installation/).
Plotting and shared-model training do not require Julia.

## Prepare Your Data

Follow the exact schemas and directory structure in [data/README.md](data/README.md).
Inputs can be kept in the repository's ignored `data/` directory or outside the
repository. To use an external directory on Linux/macOS:

```bash
export SYMBOLIC_PR_EOS_DATA_DIR="/absolute/path/to/private_inputs"
```

In PowerShell:

```powershell
$env:SYMBOLIC_PR_EOS_DATA_DIR = "C:\path\to\private_inputs"
```

Set the variable before starting Jupyter or importing the Python modules.
The fixed calibrated BIP values are in `configs/calibrated_bip.json`. Baseline
recalculation uses these values; it does not refit BIP parameters.

## Notebook Workflow

| Notebook | Purpose |
|---|---|
| `notebooks/00_shared_model.ipynb` | Joint training across all six systems and plotting |
| `notebooks/C5N2_discovery_and_vle.ipynb` | Pentane discovery and VLE panels |
| `notebooks/C6N2_discovery_and_vle.ipynb` | Hexane workflow |
| `notebooks/C7N2_discovery_and_vle.ipynb` | Heptane workflow |
| `notebooks/C9N2_discovery_and_vle.ipynb` | Nonane workflow |
| `notebooks/C10N2_discovery_and_vle.ipynb` | Decane workflow |
| `notebooks/C12N2_discovery_and_vle.ipynb` | Dodecane workflow |

Prepare inputs before executing the data-reading cells. All outputs are cleared.
Expensive operations are opt-in:

- Set `RUN_DISCOVERY = True` in a system notebook to run its PySR searches.
- Set `RUN_TRAINING = True` in the shared notebook to train the proposed model.
- Set `RECOMPUTE_BASELINES = True` to calculate new EOS baselines.
- Set `PREDICTIONS` to the generated run's `predictions.csv` before plotting.

The proposed model is trained jointly, not independently six times. Running
system searches does not automatically rewrite the shared library: the recurring
expression structures selected for the method are explicitly defined in
`script/shared_model.py`.

## Command-Line Workflow

Run commands from the repository root after preparing the input files.

### Discover expressions

```bash
python -m script.workflow discover --system C7N2 --seed 1
```

Search settings are in `configs/search_settings.json`. C5/C7/C9 use 150
iterations and population size 600; C6/C10/C12 use 120 and 800. There are 31
populations. Maximum sizes are 12, 10, 14, 12, 24 and 10, respectively.
All systems allow basic arithmetic; C12 also allows power. Unary operators
are specified per system. The loss is normalized MSE without an added parsimony
penalty. Constant optimization uses BFGS, at most eight iterations, and two
additional starts; PySR uses Newton's method for single-constant candidates.
Selection uses the largest-complexity candidate retained on the final front
within the configured size limit. Fixed seeds do not guarantee identical
parallel searches across machines. Historical C5 repetition rules are preserved
in `script/discovery_weights.py`.

### Train the shared model

```bash
python -m script.workflow train-shared --epochs 2500 --lr 5e-3 --seed 42
```

New run directories contain predictions, actual settings, library versions,
PyTorch weights, effective basis constants, normalization constants, coefficient
values and fifth-degree coefficient polynomials. They are ignored by Git and
must not be published automatically: training tables can contain measured values.

The coefficient network has two tanh hidden layers of width 12. Separate pressure
and composition models use Adam, normalized MSE, equal state weights, a penalty
`1e-4 * mean(alpha**2)`, weight decay `1e-6`, and the entire training subset at
each epoch. The physical residual is recovered by inverse normalization and
added to the calibrated baseline. Pressure is clipped below at zero and vapor
composition to `[0, 1]`.

This code retains the development workflow: discovery and normalization use all
available cached states, and a pointwise 80/20 holdout is monitored for parameter
selection. **It is not an untouched independent test set.** Pressure/composition
split seeds are 42/43 by default. A new independent evaluation should split data
before structure selection and scaling, and use a separate validation set.

### Recalculate baselines and plot

```bash
python -m script.workflow baselines --model zero --system C5N2
python -m script.workflow baselines --model calibrated --system C5N2
python -m script.workflow baselines --model generalized --system C5N2
python -m script.workflow plot --predictions outputs/<run_directory>/predictions.csv
```

Each baseline calculation writes a new directory and a solver report, retaining
failed states as NaN. Place the resulting files in the documented cache locations,
or use `BASELINE_DIRS` in a notebook. Plotting writes one PDF/PNG per system, with
separate temperature panels, and an exclusion log.

PCHIP interpolation is applied separately to liquid and vapor branches, without
extrapolation beyond sampled pressures. Baselines are aligned to available
proposed-model states. For visualization, traces break at nonfinite/unphysical
values, `abs(y-x) <= 1e-6`, or downward pressure jumps greater than 10% along
increasing liquid composition. These are plotting safeguards, not rigorous
phase-stability or criticality tests. Exclusions are logged and do not modify
source predictions. The correction requires an available PR-EOS baseline.

## Synthetic Software Demonstration

The following generates **fictional observations, fictional properties and
illustrative prediction traces** from analytic rules. No real measurements are
used. Baseline demo traces are not EOS solutions, and proposed demo traces are
not learned predictions. They cannot support scientific performance claims.

```bash
python -m script.synthetic_demo
export SYMBOLIC_PR_EOS_DATA_DIR="$PWD/outputs/synthetic_demo/data"
python -m script.workflow plot --predictions outputs/synthetic_demo/proposed_predictions.csv
python -m script.workflow train-shared --epochs 20
```

PowerShell equivalent for the environment variable:

```powershell
$env:SYMBOLIC_PR_EOS_DATA_DIR = "$PWD\outputs\synthetic_demo\data"
```

Synthetic plots have a `SYNTHETIC DEMO` title and a `Synthetic observations`
legend entry. The generator refuses to overwrite an existing demo directory;
use `--out` to choose a new one. Unset the environment variable and restart the
kernel before switching to scientific inputs. The command creates local files,
not files shipped with this release.

## Checks and Layout

```bash
python -m unittest discover -s tests -v
```

Tests create temporary fictional inputs; no external measurements are needed.
See [VALIDATION.md](VALIDATION.md) for checks performed on this code-only copy.

```text
configs/       Method settings and fixed BIP values
notebooks/     Six system notebooks and one shared-model notebook
script/        Data access, discovery, training, EOS and plotting functions
script/synthetic_demo.py   Fictional-input generator
data/README.md            Input schemas only; no dataset
tests/                    Synthetic software tests
```

## Original Data Sources

Users obtain data independently and must observe the applicable terms.
Citations do not grant redistribution permission.

| System | Original source |
|---|---|
| C5 + N2 | Silva-Oliver et al. (2006), [Fluid Phase Equilibria 250, 37-48](https://doi.org/10.1016/j.fluid.2006.09.018) |
| C6 + N2 | Eliosa-Jimenez et al. (2007), [J. Chem. Eng. Data 52, 395-404](https://doi.org/10.1021/je060341d) |
| C7 + N2 | Garcia-Sanchez et al. (2007), [J. Chem. Thermodynamics 39, 893-905](https://doi.org/10.1016/j.jct.2006.11.007) |
| C9 + N2 | Silva-Oliver et al. (2007), [J. Supercritical Fluids 42, 36-47](https://doi.org/10.1016/j.supflu.2007.01.006) |
| C10 + N2 | Garcia-Sanchez et al. (2009), [J. Chem. Eng. Data 54, 1560-1568](https://doi.org/10.1021/je800881t) |
| C12 + N2 | Garcia-Cordova et al. (2011), [J. Chem. Eng. Data 56, 1555-1564](https://doi.org/10.1021/je1012372) |

Generalized BIP: Yang, T., Siepmann, J. I., and Wu, J. (2022),
*A simple model with wide applicability for the determination of binary interaction
parameters for mixtures of n-alkanes with carbon dioxide and nitrogen*,
Industrial & Engineering Chemistry Research 61, 12229-12238.
[DOI: 10.1021/acs.iecr.2c00242](https://doi.org/10.1021/acs.iecr.2c00242).

## Licensing

Removing experimental data does not determine ownership of the code.
**A code license is not yet assigned.** The copyright holders and relevant
institution must approve the chosen license before it is added. See
[NOTICE.md](NOTICE.md). No third-party library source or proprietary font is bundled.
