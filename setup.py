import codecs
import os

from Cython.Build import cythonize
from Cython.Distutils import build_ext
from setuptools import setup, find_packages, Extension

from src.pyoteapp import version  # Edit this file to change version number

###################################################################

VERSION = version.version()

NAME = "pyote"
PACKAGES = find_packages(where="src")

for pkg in PACKAGES:
    print('package found: ' + str(pkg))

KEYWORDS = ["desktop app", "asteroid occultation timing extraction"]
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "Natural Language :: English",
    "License :: OSI Approved :: MIT License",
    "Operating System :: MacOS :: MacOS X",
    "Operating System :: Microsoft :: Windows",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Cython",
    "Programming Language :: Python :: Implementation :: CPython",
    "Topic :: Scientific/Engineering",
]
INSTALL_REQUIRES = ['pyqtgraph']

###################################################################

HERE = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    """
    Build an absolute path from *parts* and and return the contents of the
    resulting file.  Assume UTF-8 encoding.
    """
    with codecs.open(os.path.join(HERE, *parts), "rb", "utf-8") as f:
        return f.read()


extensions = [
    Extension(name='pyoteapp.c_functions',  # using dots! to get .so in correct directory
              sources=['src/pyoteapp/c_functions.pyx'])
    ]


if __name__ == "__main__":
    setup(
        name='pyote',
        ext_modules=cythonize(extensions),
        cmdclass={'build_ext': build_ext},
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
        include_package_data=True,
        classifiers=CLASSIFIERS,
        install_requires=INSTALL_REQUIRES,
    )
