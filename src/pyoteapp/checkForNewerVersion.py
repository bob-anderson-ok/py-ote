def getMostRecentVersionOfPyote():

    import subprocess

    # The call to pip that follows utilizes a trick: when pip is given a valid package but an
    # invalid version number, it writes to stderr an error message that contains a list of
    # all available versions.
    # Below is an example capture...

    # Could not find a version that satisfies the requirement
    #   pyote==?? (from versions: 1.11, 1.12, 1.13, 1.14, 1.15, 1.16)

    resp = subprocess.run(['python', '-m', 'pip', 'install', 'pyote==??'],
                          stderr=subprocess.PIPE, stdout=subprocess.PIPE)

    # Convert the byte array to a string and split into lines
    ans = resp.stderr.decode("utf-8").split('\n')

    # Split the first line of the response into its sub-strings
    ans = ans[0].split()

    if ans[0] == 'Retrying':
        return False, 'No Internet connection --- could not reach PyPI'
    elif ans[0] != 'Could':
        return False, 'Failed to find pyote package in PyPI repository'
    else:
        versionFound = ans[-1][0:-1]  # Use last string, but not the trailing right paren
        return True, versionFound


def upgradePyote():

    import subprocess

    resp = subprocess.run(['python', '-m', 'pip', 'install', '--upgrade', 'pyote'],
                          stderr=subprocess.PIPE, stdout=subprocess.PIPE)

    ans = resp.stderr.decode("utf-8").split('\n')

    return ans

if __name__ == '__main__':
    result = getMostRecentVersionOfPyote()
    print(result)
