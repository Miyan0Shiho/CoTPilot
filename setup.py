from setuptools import setup, find_packages

setup(
    name="cot_pilot",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "rich",
        "openai",
        "numpy",
        "scikit-learn"
    ],
)
