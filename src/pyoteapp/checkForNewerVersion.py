# def upgradePyote(pyoteversion):
#
#     import subprocess
#
#     try:
#         import pipenv
#         resp = subprocess.run(['python3', '-m', 'pipenv', 'install', pyoteversion],
#                               stderr=subprocess.PIPE, stdout=subprocess.PIPE)
#     except Exception:
#         resp = subprocess.run(['python', '-m', 'pip', 'install', '--user', '--upgrade', pyoteversion],
#                               stderr=subprocess.PIPE, stdout=subprocess.PIPE)
#
#     return resp.stdout.decode("utf-8").split('\n')


# The following fuction was added 23 July 2022 when (apparently) the PyPI JSON API broke.
# This uses a reliable, supported technique to get info (we are only interested in Version: )
# about a package on PyPI
def getLatestPackageVersion(package_name: str) -> str:
    import subprocess
    try:
        response = subprocess.run(['python', '-m', 'pip', 'install', f"{package_name}==0.0.0"],
                                  stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        errorResponse = response.stderr.decode("utf-8")
        versions = errorResponse.split('versions: ')
    except FileNotFoundError as e:
        print(f'{e}')
        versions = [0]
    if len(versions) == 1:  # Because the split above failed
        print('python not used to start pyote or no internet connection')
        try:
            response = subprocess.run(['python3', '-m', 'pip', 'install', f"{package_name}==0.0.0"],
                                      stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            errorResponse = response.stderr.decode("utf-8")
            versions = errorResponse.split('versions: ')
        except FileNotFoundError as e:
            print(f'{e}')
            versions = [0]
        if len(versions) == 1:
            print('python3 not used to start pyote or no internet connection')
            try:
                response = subprocess.run(['py', '-m', 'pip', 'install', f"{package_name}==0.0.0"],
                                          stderr=subprocess.PIPE, stdout=subprocess.PIPE)
                errorResponse = response.stderr.decode("utf-8")
                versions = errorResponse.split('versions: ')
            except FileNotFoundError as e:
                print(f'{e}')
                versions = [0]
            if len(versions) == 1:
                print('py not used to start pyote or no internet connection')
                # Failed to make Internet connection
                return 'Failed to connect to PyPI - Internet connection problem?'
    versions = versions[1].split(')')[0]  # Remove everything at and after ')'
    latestVersion = versions.split(',')[-1].strip()

    return latestVersion
