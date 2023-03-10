"""Package setup file."""

import re
from pathlib import Path

from setuptools import find_packages, setup

# Core package components and metadata

NAME = "tboost"
EMAIL = "btcross26@yahoo.com"
PACKAGES = find_packages()
KEYWORDS = [""]
DESCRIPTION = "General boosting framework for any regression estimator using pytorch"
LONG = """
:code:`tboost` is an implementation of the :code:`genestboost` package using pytorch
tensors rather than numpy arrays. This makes it easier to combine boosting with neural
network concepts without having to constantly move arrays back and forth between
:code:`numpy` and :code:`pytorch`. The fact that :code:`pytorch` is used also makes
it possible to use GPUs for computations.
"""

PROJECT_URLS = {
    "Documentation": ("https://btcross26.github.io/tboost/build/html/index.html"),
    "Bug Tracker": "https://github.com/btcross26/tboost/issues",
    "Source Code": "https://github.com/btcross26/tboost",
}

CLASSIFIERS = [
    "Natural Language :: English",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
]

INSTALL_REQUIRES = [
    "torch==1.*,>=1.4.0",
    "numpy==1.*,>=1.18.5",
    "scipy==1.*,>=1.4.1",
]

EXTRAS_REQUIRE = {
    "docs": ["sphinx", "sphinx_rtd_theme", "sphinx-autodoc-typehints", "recommonmark"],
    "tests": ["coverage", "mypy", "pytest", "pytest-cov", "toml"],
    "qa": [
        "pre-commit",
        "black",
        "mypy",
        "tox",
        "check-manifest",
        "flake8",
        "flake8-docstrings",
    ],
    "build": ["twine", "wheel"],
    "notebooks": [
        "jupyter",
        "ipykernel",
        "matplotlib==3.*,>=3.2.1",
        "pandas==1.*,>=1.0.4",
        "scikit-learn==0.*,>=0.22",
        "patsy==0.5.1",
    ],
}

EXTRAS_REQUIRE["dev"] = (
    EXTRAS_REQUIRE["tests"]
    + EXTRAS_REQUIRE["docs"]
    + EXTRAS_REQUIRE["qa"]
    + EXTRAS_REQUIRE["build"]
    + EXTRAS_REQUIRE["notebooks"]
)


HERE = Path(__file__).absolute().parent
PACKAGE_INIT = HERE / "tboost" / "__init__.py"
VERSION = re.match(
    r".*__version__ *= *\"([\w\.-]+)*?\".*", PACKAGE_INIT.read_text(), re.DOTALL
).group(1)
URL = PROJECT_URLS["Source Code"]
AUTHORS = (HERE / "AUTHORS").read_text().split("\n")
LICENSE = "BSD 3-Clause"


def install_pkg():
    """Configure the setup for the package."""
    setup(
        name=NAME,
        version=VERSION,
        description=DESCRIPTION,
        long_description=LONG,
        long_description_content_type="text/x-rst",
        url=URL,
        project_urls=PROJECT_URLS,
        author=AUTHORS[0],
        author_email=EMAIL,
        maintainer=AUTHORS[0],
        license=LICENSE,
        python_requires=">=3.8.0,<3.11",
        packages=PACKAGES,
        install_requires=INSTALL_REQUIRES,
        classifiers=CLASSIFIERS,
        extras_require=EXTRAS_REQUIRE,
        include_package_data=True,
        zip_safe=False,
    )


if __name__ == "__main__":
    install_pkg()
