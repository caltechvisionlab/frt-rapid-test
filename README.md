# A Rapid Test for Accuracy and Bias of Face Recognition Technology (WACV2025)

[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/abs/2502.14996)

This module benchmarks face recognition APIs (e.g., AWS Rekognition, Face++, Luxand, or Tencent) for overall performance and racial/gender bias. The benchmark queries images from internet sources and uses an unsupervised algorithm to estimate identities, so no manual labels are required. Images can be processed "on-the-fly" (in-memory) and there is no need to create or persistently store datasets.

This package was developed by Ethan Mann (emann@alumni.caltech.edu), Ignacio Serna ([ignacioserna](https://ignacioserna.github.io/)), Manuel Knott (manuel.knott@alumni.ethz.ch), and Professor Pietro Perona ([vision.caltech.edu](http://www.vision.caltech.edu/)).

---

## Setup APIs

See [docs/api-setups.md](docs/api_setups.md) for more information on setting up accounts with different cloud providers.
Once you have obtained API credentials, add them to the [scrapers.yaml](scrapers.yaml) and [services.yaml](services.yaml) files .

---

## Installing the package

### Python version requirement

This package was developed and tested with Python versions [3.9](https://www.python.org/downloads/release/python-3918/) and [3.10](https://www.python.org/downloads/release/python-31012/). If Python 3.9/3.10 is installed, you can create a virtual environment using [`virtualenv`](https://virtualenv.pypa.io/en/latest/index.html).

### pip installation

The library is [pip installable](https://pip.pypa.io/en/stable/cli/pip_install/) but not from the public index of packages. Prior to public release, you can install the package from the source code or the GitHub url. 

**Option 1.** Local pip install:
```
# create the virtual environment
virtualenv venv -p 3.9  # or 3.10

# enter the virtual environment
source venv/bin/activate

# clone the repo
git clone git@github.com:caltechvisionlab/frt-rapid-test.git

# pip install from the local repo
pip3 install frt-rapid-test/
```

**Option 2.** GitHub pip install (the repo is private, so first [make an access token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token)):
```
# create the virtual environment
virtualenv venv -p 3.9  # or 3.10

# enter the virtual environment
source venv/bin/activate

# pip install from the github url (replace "TOKEN" with your access token)
pip3 install "git+https://TOKEN@github.com/caltechvisionlab/frt-rapid-test.git"
```

**Option 3.** Clone the repository
```
# clone the repo
git clone git@github.com:caltechvisionlab/frt-rapid-test.git

# cd into the repo
cd frt-rapid-test

# create the virtual environment
virtualenv venv -p 3.9  # or 3.10

# enter the virtual environment
source venv/bin/activate

# install dependencies
pip install --upgrade pip && pip install -r requirements.txt
```

---

## Quickstart (Running the demo)

To run a quick demostration of the package, make sure to clone the repository (Option 3 above) and have your API credentials setup (see section above).
If you like, you can edit the list of names in `demo/names.csv` to scrape images of different people.

Then, run the demo script by calling the following command from the root directory of the repository:
```
python3 -m demo
```

We recommend to have a look at [demo/\_\_main\_\_.py](demo/__main__.py) for a demonstration how to use the package as a python module.

---

## Directory contents

### cfro/

The `cfro` package is the end-to-end pipeline for the facial recognition benchmark. This pipeline can run on a local OS or a Google Colab notebook with Drive mounted.

### demo/

This is a demo script to teach how to use the package. Run `python3 -m demo`.

### docs/

Contains additional documentation.

### paper-supplement/

Contains supplementary files for the published paper.

---

## License

This project is licensed under the terms of the [Apache-2.0 license](LICENSE).

---

## Citation

If you find this package useful, please consider citing our paper:

```
@InProceedings{knott_2025_rapid,
    author    = {Knott, Manuel and Serna, Ignacio and Mann, Ethan and Perona, Pietro},
    title     = {A Rapid Test for Accuracy and Bias of Face Recognition Technology},
    booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025},
    pages     = {7731-7740}
}
}
```
