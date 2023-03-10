# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config
# apidoc command: sphinx-apidoc \
#                   -d 2 \
#                   -f \
#                   -e \
#                   -M \
#                   -H genestboost \
#                   -A Benjamin\ Cross \
#                   -o ./ \
#                   ../ \
#                   ../setup*

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#


import os
import re
import sys
from pathlib import Path

from recommonmark.parser import CommonMarkParser

sys.path.insert(0, os.path.abspath("../.."))


# HELPER functions
def skip(app, what, name, obj, would_skip, options):
    """Document __init__ class method."""
    if name == "__init__":
        return False
    return would_skip


def setup(app):
    """Implement the above skip method."""
    app.connect("autodoc-skip-member", skip)


def dedupe_submodules():
    """
    Clear out duplicate submodules.
    """
    dedupe_list = [
        "genestboost.link_functions.rst",
        "genestboost.loss_functions.rst",
        "genestboost.utils.rst",
        "genestboost.weak_learners.rst",
        "genestboost.rst",
    ]

    for fn in dedupe_list:
        with open(fn, "rt") as f:
            generated_content = f.read()

        new_content = re.sub(
            "Submodules.*\nModule contents\n-*\n",
            "",
            generated_content,
            flags=re.DOTALL,
        )

        with open(fn, "wt") as f:
            f.write(new_content)


# -- Project information -----------------------------------------------------

REPO = Path(__file__).absolute().parent.parent.parent  # repo path from conf.py
PACKAGE_INIT = REPO / "genestboost" / "__init__.py"
project = "genestboost"
copyright = "2021, Benjamin Cross"
author = "Benjamin Cross"
version = re.match(
    r".*__version__ *= *\"([\w\.-]+)*?\".*", PACKAGE_INIT.read_text(), re.DOTALL
).group(1)


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.intersphinx",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# source parsers
source_parsers = {
    ".md": CommonMarkParser,
}
source_suffix = [".rst", ".md"]

# masterdoc
master_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "titles_only": False,
    "includehidden": True,
    "navigation_depth": 4,
    "canonical_url": "",
    "analytics_id": "",
    "logo_only": False,
    "sticky_navigation": True,
    "collapse_navigation": True,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": True,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# sidebar templates
html_sidebars = {
    "**": [
        "about.html",
        "navigation.html",
        "relations.html",
        "searchbox.html",
    ]
}

# -- Extension configuration -------------------------------------------------
autodoc_typehints = "none"
typehints_document_rtype = False

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

dedupe_submodules()
