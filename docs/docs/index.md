# Documentation

This section explains how the code works and how to use it.

This is the current folder structure:

```
.
├── .env
├── config.yaml
├── CONTRIBUTING.md
├── data
│   └── all_active_phase_data.csv
├── docs
│   ├── docs
│   │   └── index.md
│   ├── index.md
│   └── notes
│       └── index.md
├── environment.yaml
├── FFT-project.code-workspace
├── LICENSE
├── mkdocs.yml
├── pyproject.toml
├── README.md
├── requirements.txt
├── scripts
│   ├── workstream-1-main.py
│   └── workstream-2-main.py
├── src
│   └── fft_project
│       ├── __init__.py
│       └── config.py
└── testFileEm.py
```
To run the code

```
python scripts/workstream-1-main.py [options]]
```

### Options

- `--config <file>` - Configuration file path (default: `config.yaml`)
- `--loglevel <level>` - Set logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` (default: `INFO`)
- `--run-id <id>` - Unique run identifier (default: auto-generated timestamp)

## Output folder structure

Each run creates its own folder in the location pointed to by REMOTE_DRIVE
with the following structure

```
.
└── run-id
    ├── 1-inputs
    ├── 2-generated-data
    ├── 3-analysis
    └── 4-visualizations
```
## Example

Some inline math: $E = mc^2$.

A display equation:

$$
\nabla^2 \psi = \frac{1}{c^2}\frac{\partial^2 \psi}{\partial t^2}
$$

