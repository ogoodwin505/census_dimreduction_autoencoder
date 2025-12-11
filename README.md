<div align="center">

#  Deep Learning Composite Neighbourhood Structure Using Autoencoders

**FigShare:** [https://figshare.com/s/7aeba52e2b0af082d22d](https://figshare.com/s/7aeba52e2b0af082d22d)

<a href="https://www.python.org"><img alt="Python" src="https://img.shields.io/badge/python%20-%2314354C.svg?&style=for-the-badge&logo=python&logoColor=white"/></a>


</div>

This repository contains tools and notebooks for fully reproducing the results of the paper "Deep Learning Composite Neighbourhood Structure Using Autoencoders"

## Abstract

Dimensionality reduction techniques are a foundational component of spatial data analysis, enabling researchers to distil complex and high-dimensional datasets into lower dimensional forms for mapping and modelling of spatial processes. Principal Component Analysis (PCA), a linear dimensionality reduction method, has historically dominated this analytical approach. However, PCA’s inherent linearity restricts its effectiveness in accurately characterising the structure of spatial datasets that exhibit pronounced nonlinear relationships. In response to these limitations, this paper introduces and evaluates the utility of autoencoders (AEs), which can be configured as a nonlinear dimensionality reduction method by leveraging deep neural networks, as a robust alternative capable of better uncovering composite neighbourhood structure. To demonstrate the comparative advantage and practical utility of autoencoders, we present a case study applying this technique to small area level data derived from the 2021 Census for England and Wales. Our analysis assesses how effectively autoencoders summarise spatial variability and uncover composite dimensions of residential structure, relative to traditional linear methods. 

## Project Structure

```
├── 📄 README.md
├── 📁 data #all input files and outputs for paper
│   ├── 📁 AE_outputs
│   ├── 📁 OAC
│   ├── 📁 census_data
│   ├── 📁 geofiles
│   └── 📁 plots
├── 📁 notebooks #notebooks for reproducing results 
│   ├── 📄 1_preparedata.ipynb
│   ├── 📄 2_run_AE.ipynb
│   ├── 📄 3_ploterror_vs_dim.ipynb
│   ├── 📄 4_error_by_geo.ipynb
│   └── 📄 5_analyse_specific_oa.ipynb
├── 📄 requirements.txt #minimum set of requirements
└── 📄 uv.lock #to fully replicate the python enviroment used for the paper

```

## Installation

1. Clone the repository or download from Figshare:
   ```bash
   cd census-dimreduction-autoencoder

2. Install dependencies using a virtual enviroment and pip or UV
For pip
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    pip install -r requirements.txt

## Reproduce Results
To reproduce the full result run each notebook in order:

this will perform all stages of the analysis; 
1. Preform the required transformations and data cleaning of the input data set (England and Wales 2021 Census Data).
2. Train Autoencoder models on the Census data and produce a PCA decompostion for comparison.
3. Produce intial plots comparing the performance for different dimensions (Fig 3)
4. Produce values, plots and maps analysing the geographic distubutions of the errors (Figs 4-7)
5. Analyse specific Output Areas for results

Each of the notebooks can be run in isolation, with the outputs of each stage being saved to /data. These outputs are all available in the figshare data folder.
Stage 2 is time consuming to run (several hours depending on PC) as it trains Autoencoder models for a varity of parameter points.

## Data Sources

All data required to repreduce the results is included in the figshare data directory.

The original sources are;

England and Wales Census Data - [download](https://www.nomisweb.co.uk/sources/census_2021_bulk)
Geofiles and boundaries - [download](https://geoportal.statistics.gov.uk)
IMD - [download](https://data.geods.ac.uk/dataset/index-of-multiple-deprivation-imd)
OAC - [download](https://data.geods.ac.uk/dataset/output-area-classification-2021)

