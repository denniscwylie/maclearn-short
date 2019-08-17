# Principles of Machine Learning for Bioinformatics

This course introduces some of the basic concepts and tools of machine
learning. Topics covered include: unsupervised learning and
dimensionality reduction, feature selection, and supervised learning
methods for classification (e.g., kNN, logistic regression, SVM,
etc.).

Lecture notes are provided in the slide deck
[maclearn.pdf](maclearn.pdf).

The directories **microarray**, **pcr**, and **rnaseq** contain
example data sets. Most of the remaining files in the repository are R
or python scripts.

## Suggested prerequisites

Recommended for students with some prior knowledge of either R or
python. **Participants are expected to provide their own laptops with
recent versions of R and/or python installed.** Use of the included
scripts will require installation of several free software packages
(including R packages and/or python libraries such as including pandas
and sklearn).

## R packages

### from CRAN

The command below can be run within an R session to install most of
the required packages from CRAN; **some of these may take a while to
install, recommend installation prior to class if you intend to run
the R scripts.**

```R
install.packages(c('caret', 'e1071', 'ggplot2', 'ggrepel',
                   'glmnet', 'MASS', 'matrixStats', 'rgl'))
```
NOTE: the package **rgl** is only used in generating a 3d plot during
the explanation of principal components analysis (PCA); feel free to
remove it from the list of packages if you have trouble with the
installation.

### from Bioconductor

The package **genefilter** can be installed from Bioconductor using the
following code again run within an R session.

```R
install.packages('BiocManager')
BiocManager::install('genefilter')
```

### from github

The package **sparsediscrim** can be installed from github using the
following code again run within an R session
```R
install.packages('devtools') # if devtools not already installed
devtools::install_github('ramhiser/sparsediscrim')
```
NOTE: **sparsediscrim** contains an implementation of a naive Bayes
method (DLDA) which we will not have time to discuss in the short
version of this course but which can be quite useful.

## Python modules

The following Python modules are used in the included scripts; again I
would **recommend installing prior to class if you intend to run the
Python scripts**:
- numpy
- pandas
- scikit-learn
- matplotlib
- plotnine

## Scripts

| R                                    | Python                                 | Notes                                                  |
|--------------------------------------|----------------------------------------|--------------------------------------------------------|
| [pca.R](pca.R)                       | [pca.py](pca.py)                       |                                                        |
| [knn\_sim.R](knn\_sim.R)             | [knn\_sim.py](knn\_sim.py)             | compare resub vs. test performance on simulated data   |
| [knn\_sim\_cv.R](knn\_sim\_cv.R)     | [knn\_sim\_cv.py](knn\_sim\_cv.py)     | show cross-validation (cv) removes resub bias          |
| [knn\_real.R](knn\_real.R)           | [knn\_real.py](knn\_real.py)           | t-test feature selection/extraction + knn on real data |
| [logistic\_real.R](logistic\_real.R) | [logistic\_real.py](logistic\_real.py) |                                                        |
| [svm\_real.R](svm\_real.R)           | [svm\_real.py](svm\_real.py)           |                                                        |
