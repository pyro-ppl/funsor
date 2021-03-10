# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import glob
import os
import shutil
import sys

import sphinx_rtd_theme

# import pkg_resources

# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------

project = u"Funsor"
copyright = u"2019, Uber Technologies, Inc"
author = u"Uber AI Labs"

# The short X.Y version
version = u"0.0"

if "READTHEDOCS" not in os.environ:
    # if developing locally, use funsor.__version__ as version
    from funsor import __version__  # noqaE402

    version = __version__

    html_context = {"github_version": "master"}

# The full version, including alpha/beta/rc tags
release = version


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_gallery.gen_gallery",
]

# Disable documentation inheritance so as to avoid inheriting
# docstrings in a different format, e.g. when the parent class
# is a PyTorch class.

autodoc_inherit_docstrings = False
autodoc_member_order = "bysource"

# FIXME the sphinx version on readthedocs does not
# support this option. These must be manually added.
# autodoc_default_options = {
#     'member-order': 'bysource',
#     'show-inheritance': True,
#     'special-members': True,
#     'undoc-members': True,
#     'exclude-members': '__dict__,__module__,__weakref__',
# }

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = [".rst", ".ipynb"]

# do not execute cells
nbsphinx_execute = "never"

# Don't add .txt suffix to source files:
html_sourcelink_suffix = ""

# The master toctree document.
master_doc = "index"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path .
exclude_patterns = [".ipynb_checkpoints", "examples/*ipynb", "examples/*py"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"


# do not prepend module name to functions
add_module_names = False


# This is processed by Jinja2 and inserted before each notebook
nbsphinx_prolog = r"""
{% set docname = 'tutorials/' + env.doc2path(env.docname, base=None).split('/')[-1] %}
:github_url: https://github.com/pyro-ppl/funsor/blob/master/{{ docname }}

.. raw:: html

    <div class="admonition note">
      Interactive online version:
      <span style="white-space: nowrap;">
        <a href="https://colab.research.google.com/github/pyro-ppl/funsor/blob/{{ env.config.html_context.github_version }}/{{ docname }}">
          <img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg"
            style="vertical-align:text-bottom">
        </a>
      </span>
    </div>
"""  # noqa: E501


# -- Copy notebook files
# NB: tutorials and examples can be added to `index.rst` file using the paths
#     tutorials/foo
#     examples/foo
# without extensions .ipynb or .py
# TODO: find a solution for an example subfolder, e.g. examples/mixed_hmm folder
#
# To add thumbnail images for tutorials/examples in funsor docs, using
# .. nbgallery:: instead of .. toctree:: and add png thumnail images
# with corresponding names in _static/img/tutorials or _static/img/examples folders.
# For example, we can add minipyro.png to _static/img/examples/ folder.

if not os.path.exists("tutorials"):
    os.makedirs("tutorials")

for src_file in glob.glob("../../tutorials/*.ipynb"):
    dst_file = os.path.join("tutorials", src_file.split("/")[-1])
    shutil.copy(src_file, "tutorials/")


# -- Convert scripts to notebooks

sphinx_gallery_conf = {
    "examples_dirs": ["../../examples"],
    "gallery_dirs": ["examples"],
    # only execute files beginning with plot_
    "filename_pattern": "/plot_",
    # 'ignore_pattern': '(minipyro|__init__)',
    # not display Total running time of the script because we do not execute it
    "min_reported_time": 1,
}


# -- Add thumbnails images

nbsphinx_thumbnails = {}

for src_file in glob.glob("../../tutorials/*.ipynb") + glob.glob("../../examples/*.py"):
    toctree_path = "tutorials/" if src_file.endswith("ipynb") else "examples/"
    filename = os.path.splitext(src_file.split("/")[-1])[0]
    png_path = "_static/img/" + toctree_path + filename + ".png"
    # use Pyro logo if not exist png file
    if not os.path.exists(png_path):
        png_path = "_static/img/pyro_logo_wide.png"
    nbsphinx_thumbnails[toctree_path + filename] = png_path


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}


# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "funsordoc"


# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, "Funsor.tex", u"Funsor Documentation", u"Uber AI Labs", "manual")
]

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "Funsor", u"Funsor Documentation", [author], 1)]

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "Funsor",
        u"Funsor Documentation",
        author,
        "Funsor",
        "Functional analysis + tensors + symbolic algebra.",
        "Miscellaneous",
    )
]


# -- Extension configuration -------------------------------------------------

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "multipledispatch": ("https://multiple-dispatch.readthedocs.io/en/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "opt_einsum": ("https://optimized-einsum.readthedocs.io/en/stable/", None),
    "pyro": ("http://docs.pyro.ai/en/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "torch": ("http://pytorch.org/docs/master/", None),
}

# @jpchen's hack to get rtd builder to install latest pytorch
if "READTHEDOCS" in os.environ:
    os.system(
        "pip install torch==1.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html"
    )
    # pyro needs to be installed after torch so pyro doesnt install the bloated torch-1.0 wheel
    os.system("pip install pyro-ppl")
