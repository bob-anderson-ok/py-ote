import codecs
import os
import re

from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext


###################################################################

NAME = "pyote"
PACKAGES = find_packages(where="src")
#META_PATH = os.path.join("src", "bob__init__.py")
KEYWORDS = ["desktop app", "asteroid occultation timing extraction"]
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "Natural Language :: English",
    "License :: OSI Approved :: MIT License",
    "Operating System :: MacOS :: MacOS X",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Cython",
    "Programming Language :: Python :: Implementation :: CPython",
    "Topic :: Scientific/Engineering",
]
INSTALL_REQUIRES = ['pyqtgraph']

###################################################################

HERE = os.path.abspath(os.path.dirname(__file__))

# print('HERE: ' + HERE)
# print('META_PATH: ' + META_PATH)
# print('num pakages: ' + str(len(PACKAGES)))
# for pkg in PACKAGES:
#     print(pkg)

def read(*parts):
    """
    Build an absolute path from *parts* and and return the contents of the
    resulting file.  Assume UTF-8 encoding.
    """
    with codecs.open(os.path.join(HERE, *parts), "rb", "utf-8") as f:
        return f.read()

# META_FILE = read(META_PATH)


def find_meta(meta):
    """
    Extract __*meta*__ from META_FILE.
    """
    meta_match = re.search(
        r"^__{meta}__ = ['\"]([^'\"]*)['\"]".format(meta=meta),
        META_FILE, re.M
    )
    if meta_match:
        return meta_match.group(1)
    raise RuntimeError("Unable to find __{meta}__ string.".format(meta=meta))

extensions = [
    Extension(name='pyoteapp.c_functions',  # using dots! to get .so in correct directory
              sources=['src/pyoteapp/c_functions.pyx'])
    ]


if __name__ == "__main__":
    setup(
        name='pyote',
        ext_modules=cythonize(extensions),
        cmdclass={'build_ext': build_ext},
        description='py-ote is a simplified subset of R-OTE',
        license='License :: OSI Approved :: MIT License',
        url=r'https://github.com/bob-anderson-ok/py-ote',
        version='1.2.dev0',
        author='Bob Anderson',
        author_email='bob.anderson.ok@gmail.com',
        maintainer='Bob Anderson',
        maintainer_email='bob.anderson.ok@gmail.com',
        keywords=KEYWORDS,
        long_description=read("README.rst"),
        packages=PACKAGES,
        package_dir={"": "src"},
        zip_safe=False,
        classifiers=CLASSIFIERS,
        install_requires=INSTALL_REQUIRES,
    )
