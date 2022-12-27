# import subprocess
# import json
# import re
# import urllib.request
# import urllib.error
# import urllib.parse
# from distutils.version import StrictVersion


# 23 July 2022 It appears that the PyPI JSON API is no longer working in a way that enables
# us to find the version number of the latest pymovie in the PyPI repository, so we are
# removing it from use.
# def getMostRecentVersionOfPyOTEViaJson():
#
#     # !!!!! Many thanks to Kia Getrost for supplying this much improved version of 'getMostRecentVersionOfPymovie'
#
#     # Returns tuple of gotVersion, latestVersion
#     # (boolean and version-or-error string)
#
#     pkgName = "pyote"
#
#     # Do a JSON request to pypi to get latest version:
#     url = f"https://pypi.org/pypi/{pkgName}/json"
#     text = getUrlAsText(url)
#     if text is None:
#         return False, "Could not contact pypi.org to check for latest version"
#
#     # Parse the JSON result:
#     try:
#         data = json.loads(text)
#     except ValueError:
#         return False, "Could not parse JSON response from pypi.org"
#
#     # Sort versions to get the latest:
#     versions = sorted(data["releases"], key=StrictVersion, reverse=True)
#     latestVersion = versions[0]
#
#     # Ensure we have a seemingly valid vesrion number:
#     if not re.match(r"\d+\.\d+\.\d+", latestVersion):
#         return False, f"Garbled version `{latestVersion}' from pypi.org"
#
#     # All is well, return result:
#     return True, latestVersion
#
#
# def getUrlAsText(url):
#     # Returns text string of `url', or None on error
#     try:
#         request = urllib.request.Request(url)
#         response = urllib.request.urlopen(request)
#     except urllib.error.URLError as exception:
#         if hasattr(exception, "reason"):
#             print(f"Fetch of `{url}' failed: {exception.reason}")
#         elif hasattr(exception, "code"):
#             print(f"Fetch of `{url}' failed: returned HTTP code {exception.code}")
#         else:
#             print(f"Fetch of `{url}' failed: Unknown reason")
#         return None
#     html = response.read()
#     return html.decode("utf-8")


def upgradePyote(pyoteversion):

    import subprocess

    # noinspection PyBroadException
    try:
        import pipenv
        resp = subprocess.run(['python3', '-m', 'pipenv', 'install', pyoteversion],
                              stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    except Exception:
        resp = subprocess.run(['python', '-m', 'pip', 'install', '--user', '--upgrade', pyoteversion],
                              stderr=subprocess.PIPE, stdout=subprocess.PIPE)

    return resp.stdout.decode("utf-8").split('\n')


# The following fuction was added 23 July 2022 when (apparently) the PyPI JSON API broke.
# This uses a reliable, supported technique to get info (we are only interested in Version: )
# about a package on PyPI
def getLatestPackageVersion(package_name: str) -> str:
    import subprocess
    response = subprocess.run(['python3', '-m', 'pip', 'install', f"{package_name}==0.0.0"],
                              stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    errorResponse = response.stderr.decode("utf-8")
    versions = errorResponse.split('versions: ')
    if len(versions) == 1:  # Because the split above failed
        # Failed to make Internet connection
        return 'Failed to connect to PyPI - Internet connection problem?'
    versions = versions[1].split(')')[0]  # Remove everything at and after ')'
    latestVersion = versions.split(',')[-1].strip()

    return latestVersion
