from setuptools import setup, find_packages

setup(
    name="xrai",
    version="0.1",
    packages=find_packages(),
    install_requires=[
    "scipy",
    "statistics",
    "numpy",
    "pandas",
    "matplotlib",
    "scikit-learn",
    "gpboost",
    "xgboost",
    "shap",
    "merf",
    "openpyxl",
    "xlsxwriter",
    "featurewiz",
],
    author="Christopher Lalk",
    author_email="christopher.lalk@uni-osnabrueck.de",
    description="XRAI (eXplanable Regression-based Artificial Intelligence): integration of regression-based machine-learning and eXplainable AI via the SHAP-package",
    url="https://github.com/PsyRes-Osnabrueck-University/xrai",
)