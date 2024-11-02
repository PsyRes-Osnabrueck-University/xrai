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
    author_email="clalk@example.com",
    description="A package that reads an Excel file and sums all the columns",
    url="https://github.com/chrislalk/xrai",
)