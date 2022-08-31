# https://www.freecodecamp.org/news/how-to-create-and-upload-your-first-python-package-to-pypi/
from setuptools import setup, find_packages

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name = "frt-rapid-test",
    version = "1.0.0",
    author = "Ethan Mann, Ignacio Serna, Manuel Knott, and Pietro Perona",
    author_email = "vision@caltech.edu",
    description = "This module benchmarks face recognition APIs (e.g., AWS Rekognition or Azure Face) for overall performance and racial/gender bias. The benchmark scrapes a face dataset from news articles and uses an unsupervised algorithm to score APIs on this dataset, so no manual labels are required.",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/caltechvisionlab/frt-rapid-test",
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages = find_packages(),
    python_requires = "<3.10",
    install_requires = required
)