# A Rapid Test for Accuracy and Bias of Face Recognition Technology

Preprint available on [arXiv](TODO).

This module benchmarks face recognition APIs (e.g., AWS Rekognition or Azure Face) for overall performance and racial/gender bias. The benchmark scrapes a face dataset from internet sources and uses an unsupervised algorithm to score APIs on this dataset, so no manual labels are required.

This package was developed by Ethan Mann (emann@alumni.caltech.edu), Ignacio Serna (ignacio.serna@uam.es), Manuel Knott (mknott@ethz.ch), and Professor Pietro Perona ([vision.caltech.edu](http://www.vision.caltech.edu/)).

## Section 1a: Setup accounts with cloud providers

The package currently supports two cloud providers. Here are instructions to setup each provider. For demo purposes, it is easiest to start with Face++. The free tier of each provider is sufficient for small datasets.

### Amazon AWS
- Follow the steps [here](https://aws.amazon.com/premiumsupport/knowledge-center/create-and-activate-aws-account/) to create an account, then read the "Programmatic access" section of [this guide](https://docs.aws.amazon.com/general/latest/gr/aws-sec-cred-types.html) to get an access key ID and secret access key.

### Face++
- Create an account [here](https://console.faceplusplus.com/register) and follow [these steps](https://console.faceplusplus.com/documents/7079083) to obtain an API key and API secret.

## Section 1b: Choose your Images API

### Google News

If you choose to use Google News for image scraping, no additional steps are needed.

### Google Images

Follow the instructions [here](https://developers.google.com/custom-search/v1/overview) to create a Google Custom Search Engine (CSE) and obtain an API key. The CSE should be configured to search the entire web, not just a subset of sites.
Be aware that, Custom Search JSON API provides 100 search queries per day for free. If you need more, you may sign up for billing in the API Console. Additional requests cost $5 per 1000 queries, up to 10k queries per day.

---

## Section 2: Running the package

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

### Quickstart commands

Here are steps to run the demo experiment (skip steps 3-5 if you already pip-installed the package):
```
# 1. clone the repo
git clone git@github.com:caltechvisionlab/frt-rapid-test.git

# 2. cd into the repo
cd frt-rapid-test

# 3. create the virtual environment
virtualenv venv -p 3.9  # or 3.10

# 4. enter the virtual environment
source venv/bin/activate

# 5. install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 6. Add API credentials to scrapers.yaml and serices.yaml

# 7. (optionally) edit demo/names.csv to scrape images of different people

# 8. (optionally) edit demo__main__.py to only run specific steps of the benchmark

# 9. run demo experiment
python3 -m demo

# 10. exit the virtual environment
deactivate
```

---

## Section 3: Directory contents

### cfro/

The `cfro` package is the end-to-end pipeline for the facial recognition benchmark. This pipeline can run on a local OS or a Google Colab notebook with Drive mounted.

### demo/

This is a demo script to teach how to use the package. Run `python3 -m demo`.

### paper-supplement/

Contains supplementary files for the published paper.