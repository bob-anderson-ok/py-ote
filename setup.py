import codecs
import os

# To get the wheel build to work in python 3.7 Anaconda3 5.3.1 the follwing changes were made...
# Commented out the following line..
# from setuptools import setup, find_packages, Extension

# Added the following lines...
from setuptools import find_packages
from distutils.core import setup
# End changes

from src.pyoteapp import version  # Edit this file to change version number

###################################################################

VERSION = version.version()

NAME = "pyote"
PACKAGES = find_packages(where="src")

for pkg in PACKAGES:
    print(f'package found: {str(pkg)}')

KEYWORDS = ["desktop app", "asteroid occultation timing extraction"]
CLASSIFIERS = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Science/Research",
    "Natural Language :: English",
    "License :: OSI Approved :: MIT License",
    "Operating System :: MacOS :: MacOS X",
    "Operating System :: Microsoft :: Windows",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: Implementation :: CPython",
    "Topic :: Scientific/Engineering",
]

# This is borrowed from pymovie so that an installation of pyote following a new Anaconda3 install
# makes no complaints about what pymovie needs, that way the order if intalling pymovie and pyote won't matter
INSTALL_REQUIRES = ['pyqtgraph==0.12.4', 'opencv-python', 'astroquery', 'resource',
                    'scikit-image(>=0.15.0)', 'wheel', 'requests', 'pyephem',
                    'PyQt5', 'pandas',
                    'winshell;platform_system=="Windows"',
                    'pypiwin32;platform_system=="Windows"', 'matplotlib', 'numpy<=1.22', 'astropy', 'scikit-image',
                    'scipy', 'numba>=0.41.0', 'Adv2>=1.2.0', 'PyQt5', 'openpyxl', 'xlrd', 'xlsxwriter']

###################################################################

HERE = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    """
    Build an absolute path from *parts* and and return the contents of the
    resulting file.  Assume UTF-8 encoding.
    """
    with codecs.open(os.path.join(HERE, *parts), "rb", "utf-8") as f:
        return f.read()


if __name__ == "__main__":
    setup(
        name='pyote',
        description='pyote is a simplified subset of R-OTE',
        license='License :: OSI Approved :: MIT License',
        url=r'https://github.com/bob-anderson-ok/py-ote',
        version=VERSION,
        author='Bob Anderson',
        author_email='bob.anderson.ok@gmail.com',
        maintainer='Bob Anderson',
        maintainer_email='bob.anderson.ok@gmail.com',
        keywords=KEYWORDS,
        long_description=read("README.rst"),
        packages=PACKAGES,
        package_dir={"": "src"},
        zip_safe=False,
        package_data={'': ['*.bat','*.xlsx']},
        include_package_data=True,
        classifiers=CLASSIFIERS,
        install_requires=INSTALL_REQUIRES,
    )
