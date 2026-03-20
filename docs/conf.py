# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


project = "bayesianbandits"
copyright = "2023, Rishi Kulkarni"
author = "Rishi Kulkarni"
release = "1.2.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc.typehints",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "nbsphinx",
]
intersphinx_mapping = {
    "sklearn": ("https://scikit-learn.org/stable/", None),
}
napoleon_preprocess_types = True
napoleon_type_aliases = {
    "NDArray": "np.typing.NDArray",
    "ArrayLike": "np.typing.ArrayLike",
}
autodoc_type_aliases = {
    "NDArray": "np.typing.NDArray",
    "ArrayLike": "np.typing.ArrayLike",
}
napoleon_custom_sections = [("Subclass Parameters", "params_style")]

autosummary_generate = True

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_short_title = "bayesianbandits"
html_theme_options = {
    "logo": {
        "text": "bayesianbandits",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/rishi-kulkarni/bayesianbandits",
            "icon": "fa-brands fa-github",
        },
    ],
    "navigation_depth": 3,
    "show_toc_level": 2,
}
