def getMostRecentVersionOfPyote():

    import pip
    import sys
    import os

    # We are going to run a pip command that writes to stderr and stdout.  In order to
    # capture that output, we redirect sys.stderr and sys.stdout to our own files.  But
    # we will have to undo this redirection, so we save the current assignments...
    savedStdErr = sys.stderr
    savedStdOut = sys.stdout

    # ... and open sockets of our own ...
    fsockErr = open('bobsStdErr.log', 'w')
    fsockStd = open('bobsStdOut.log', 'w')

    # ... and do the redirection.
    sys.stderr = fsockErr
    sys.stdout = fsockStd

    # The call to pip that follows utilizes a trick: when pip is given a valid package but an
    # invalid version number, it writes to stderr an error message that contains a list of
    # all available versions.
    # Below is an example capture...

    # Could not find a version that satisfies the requirement
    #   pyote==?? (from versions: 1.11, 1.12, 1.13, 1.14, 1.15, 1.16)

    pip.main(['install', 'pyote==??'])  # Give an invalid version number in the request to force error.

    # Restore the original assignments of stderr and stdout
    sys.stderr = savedStdErr
    sys.stdout = savedStdOut

    fsockErr.close()
    fsockStd.close()

    fsockErr = open('bobsStdErr.log', 'r')
    with fsockErr:
        pipResult = fsockErr.readline()

    os.remove('bobsStdErr.log')
    os.remove('bobsStdOut.log')

    ans = pipResult.split()
    if ans[0] == 'Retrying':
        return False, 'No Internet connection --- could not reach PyPI'
    elif ans[0] != 'Could':
        return False, 'Failed to find pyote package in PyPI repository'
    else:
        versionFound = ans[-1][0:-1]  # Use last string, but not the trailing right paren
        return True, versionFound


def upgradePyote():

    import pip
    import sys
    import os

    # We are going to run a pip command that writes to stderr and stdout.  In order to
    # capture that output, we redirect sys.stderr and sys.stdout to our own files.  But
    # we will have to undo this redirection, so we save the current assignments...
    savedStdErr = sys.stderr
    savedStdOut = sys.stdout

    # ... and open sockets of our own ...
    fsockErr = open('bobsStdErr.log', 'w')
    fsockStd = open('bobsStdOut.log', 'w')

    # ... and do the redirection.
    sys.stderr = fsockErr
    sys.stdout = fsockStd

    pip.main(['install', '--upgrade', 'pyote'])

    # Restore the original assignments of stderr and stdout
    sys.stderr = savedStdErr
    sys.stdout = savedStdOut

    fsockErr.close()
    fsockStd.close()

    fsockStd = open('bobsStdOut.log', 'r')
    with fsockStd:
        pipResult = fsockStd.readlines()

    os.remove('bobsStdErr.log')
    os.remove('bobsStdOut.log')

    return pipResult

if __name__ == '__main__':
    result = getMostRecentVersionOfPyote()
    print(result)
