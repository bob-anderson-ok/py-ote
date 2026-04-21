import json
import urllib.error
from urllib.request import urlopen


def getLatestPackageVersion(package_name: str) -> str:
    """Return the latest released version of *package_name* on PyPI, or a
    human-readable error string starting with 'Failed to connect to PyPI'
    if the query could not be completed. Uses the PyPI JSON API via stdlib
    urllib so no `python`/`pip` subprocess and no extra dependency is
    required (the old subprocess-based probe printed noisy messages on
    systems where `python`/`python3`/`py` were not on PATH, e.g. uv-launched
    installs on macOS)."""
    try:
        with urlopen(f'https://pypi.org/pypi/{package_name}/json', timeout=5) as resp:
            data = json.load(resp)
        return data['info']['version']
    except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
        return f'Failed to connect to PyPI - Internet connection problem? ({e})'
