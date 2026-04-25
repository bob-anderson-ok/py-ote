import json
import re
import urllib.error
from urllib.request import Request, urlopen


def getLatestReleaseVersion(repo: str) -> str:
    """Return the tag of the latest GitHub Release of *repo* (in 'owner/name'
    form), with any leading 'v' stripped. On failure, return a string
    starting with 'Failed: ' that names the cause; callers should test for
    that prefix before parsing the result as a version.

    Hits GitHub's REST API (/repos/{repo}/releases/latest) directly via
    stdlib urllib, no extra dependency.
    """
    try:
        req = Request(
            f'https://api.github.com/repos/{repo}/releases/latest',
            headers={'Accept': 'application/vnd.github+json'},
        )
        with urlopen(req, timeout=5) as resp:
            data = json.load(resp)
        tag = data['tag_name'].strip()
        if not tag:
            return 'Failed: empty tag from GitHub'
        return tag.lstrip('vV')
    except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
        return f'Failed: GitHub Releases query failed - {e}'


def _parse_version(s: str) -> tuple:
    # Tolerate prefixes/suffixes ('v1.2.3', '1.2.3a1') by extracting leading
    # digits from each dot-separated segment. Returns () for unparseable
    # input, which sorts before any real version.
    parts = []
    for segment in s.lstrip('vV').strip().split('.'):
        m = re.match(r'\d+', segment)
        if m is None:
            break
        parts.append(int(m.group(0)))
    return tuple(parts)


def is_newer(latest: str, current: str) -> bool:
    return _parse_version(latest) > _parse_version(current)
