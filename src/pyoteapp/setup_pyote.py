# -*- coding: utf-8 -*-
#
# Run the build process by running the command
#
#   python setup-pyote.py build_ext --inplace
#
# followed by:
#
#   python setup-pyote.py build
#
# If everything works well you should find a subdirectory in the build
# subdirectory that contains the files needed to run the application

# Special things I had to do to get this to work ...
#
#   Manually copied scipy\spatial\ckdtree.py from my Anaconda distribution into
#   the scipy\spatial folder inside the build\exe.win-amd64-3.6 directory
#   Do this immediately after the initial build completes.
#
#   Copy c:\Users\Bob\Anaconda3\Library\plugins\platforms into the
#   build\exe.win-amd64-3.6 directory
#   Do this immediately after the initial build completes.
#
#   Set the following environment variables:
#       TCL_LIBRARY = c:\Users\Bob\Anaconda3\tcl\tcl8.6
#       TK_LIBRARY  = c:\Users\Bob\Anaconda3\tcl\tk8.6
#
#   Modified line 564 in the hooks.py module in the cx_Freeze folder as follows..
#       In c:\Users\Bob\Anaconda3\Lib\site-packages\cx_Freeze\hooks.py
#           changed scipy.lib to scipy._lib (note underscore)

import sys
from cx_Freeze import Executable
from cx_Freeze import setup as cx_setup
from distutils.core import setup

from Cython.Build import cythonize

# Compile the cython modules
setup(ext_modules=cythonize('c_functions.pyx'))

# For a GUI based app, this is needed
base = None
if sys.platform == 'win32':
    base = 'Win32GUI'

# List all the packages that cx_Freeze couldn't find automatically.  This list
# gets developed by iteration: try the build, see what it fails on, add the missing
# package (or module --- see the list that follows), keep at it until success arrives
packages = ['scipy',
            'pkg_resources._vendor.appdirs',
            'pkg_resources._vendor.packaging.specifiers',
            'pkg_resources._vendor.packaging.requirements',
            'pyqtgraph.debug',
            'pyqtgraph.ThreadsafeTimer'
            ]

# List the modules that cx_Freeze failed to find automatically
modules = ['numpy.core._methods',
           'numpy.lib.format',
           'scipy.__config__',
           'scipy.spatial.ckdtree'
           ]

options = {
    'build_exe': {
        'includes': modules,
        'packages': packages
    }
}

executables = [
    Executable('pyote.py', base=base)
]

cx_setup(name='SimpleOTE',
         version='1.1',
         description='Python version of R-OTE essentials',
         options=options,
         executables=executables
         )
