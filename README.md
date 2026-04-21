# PyOTE

PyOTE is a Python implementation of the essential parts of the R-OTE
occultation timing extractor. It analyses lightcurves recorded during
asteroid occultations and produces timing measurements — D and R event
times with error bars, false-positive estimates, and publication-ready
lightcurve plots.

The name was chosen out of respect and deference to R-OTE, the original
Occultation Timing Extractor that this tool builds on. PyOTE has fewer
"bells and whistles" than R-OTE, runs faster, and is easier to deploy.

## Installing

### Windows: download the executable (easiest)

Windows users can skip the Python toolchain entirely. Go to the
[latest release](https://github.com/bob-anderson-ok/py-ote/releases/latest)
page and download `PyOTE.exe` from the *Assets* section. Double-click to
run — no installation, no Python setup required. The executable bundles
its own Python runtime and all dependencies.

The first activation will take extra time because it will be downloading
the proper version of Python and all the other dependencies. Subsequent
activations will only take a few seconds.

To update later, just download the newer `PyOTE.exe` from the release
page and replace the old one.

If you'd rather run from source (all platforms), follow the uv-based
instructions below.

### Installing from source (any platform)

PyOTE uses [uv](https://docs.astral.sh/uv/) to manage its Python
environment. You do **not** need Python pre-installed — uv will
automatically download the correct version (3.10) on first run.

**Step 1.** Install uv (one line, user-scope, no admin rights required).

Windows, in PowerShell:

```
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

macOS or Linux, in a terminal:

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Close and reopen your terminal afterwards so the new `uv` command is on
your PATH.

**Step 2.** Get the PyOTE source. Either clone with git:

```
git clone https://github.com/bob-anderson-ok/py-ote.git
cd py-ote
```

or download the repository as a ZIP from the GitHub page, unzip it, and
`cd` into the extracted folder.

**Step 3.** Launch PyOTE:

```
uv run pyote
```

On first run, uv downloads Python 3.10 (if not already present), installs
the pinned dependencies from `uv.lock` into a local `.venv` folder, and
opens the PyOTE window. Subsequent runs are near-instant.

### Updating

To pick up a new PyOTE release:

```
git pull
uv run pyote
```

uv automatically re-syncs the environment whenever `uv.lock` has changed,
so there is nothing else to do.

### Troubleshooting

* **"uv: command not found"** — close and reopen your terminal, or follow
  the PATH instructions printed by the uv installer.
* **Windows SmartScreen warning on the uv installer** — click *More info*
  → *Run anyway*. The installer is published by Astral.
* **Corporate proxy / firewall issues** — uv honours standard
  `HTTPS_PROXY` and `HTTP_PROXY` environment variables.
