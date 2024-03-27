# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'TELF'
copyright = '2022, LANL'
author = 'Maksim E. Eren, Nicholas Solovyev, Ryan Barron, Manish Bhattarai, Ismael Boureima, Erik Skau, Kim Rasmussen, Boian S. Alexandrov'
release = '0.0.12'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx_automodapi.automodapi",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
]
autoclass_content = 'both'
bibtex_bibfiles = ['refs.bib']

templates_path = ['_templates']
exclude_patterns = ['.ipynb_checkpoints', "version.py"]



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
