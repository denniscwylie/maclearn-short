#!/usr/bin/env Rscript

library(rgl)
library(matrixStats)

source('maclearn_util.R')

x = read.table('rnaseq/shen2012/19-tissues-expr.tsv.gz',
               sep='\t', header=TRUE, row.names=1, check.names=FALSE)

## extract tissue types from colnames of x
tissue = gsub('\\d*(-.*)?', '', colnames(x))
names(tissue) = colnames(x)
## define coarser division of tissue types for coloring plots
tissueCoarse = structure(factor(c(
    boneMarrow = 'lymphatic',
    brain = 'nervous',
    cerebellum = 'nervous',
    cortex = 'nervous',
    heart = 'circulatory',
    intestine = 'digestive/excretory',
    kidney = 'digestive/excretory',
    limb = 'other',
    liver = 'digestive/excretory',
    lung = 'respiratory',
    mef = 'other',
    mESC = 'other',
    olfactory = 'nervous',
    placenta = 'other',
    spleen = 'lymphatic',
    testes = 'other',
    thymus = 'lymphatic'
)[tissue]), names=names(tissue))
## these are the colors we will use:
colVals = c(c(
    circulatory = 'firebrick3',
    'digestive/excretory' = 'goldenrod',
    lymphatic = 'lightseagreen',
    nervous = 'darkorchid2',
    other = 'darkslategray',
    respiratory = 'dodgerblue'
)[tissueCoarse], rep('black', 3))

## look at only three genes first...
x3 = data.frame(t(x[c('NM_008084', 'NM_008221', 'NM_011428'), ]))
## sweep out genewise-means
x3 = sweep(x3, 2, colMeans(x3), `-`)
## let's use better names for these three genes:
colnames(x3) = c('Gapdh', 'Hbb-y', 'Snap25')
## save original version of x3 as x30
x30 = x3

## add one row for each gene to data.frame x3 for plotting purposes
x3txt = data.frame(diag(log2(10), 3)); colnames(x3txt) = colnames(x3)
x3 = rbind(x3, x3txt)
textVals = c(rep('', length(colVals)-3), colnames(x3))


## =============================================================================
## 3d PCA plot:
## use rgl for 3d plotting of data in x3 data.frame:
open3d()
plot3d(x3[ , 1], x3[ , 2], x3[ , 3],
       col=colVals, add=TRUE,
       type='s', box=FALSE, axes=FALSE, radius=0.15,
       xlab='', ylab='', zlab='')
text3d(x3[ , 1], x3[ , 2], x3[ , 3],
       texts=textVals, adj=1.15, col='black')
segments3d(c(0, 3.321928), c(0, 0), c(0, 0), col='black')
segments3d(c(0, 0), c(0, 3.321928), c(0, 0), col='black')
segments3d(c(0, 0), c(0, 0), c(0, 3.321928), col='black')

## do principal components analysis (PCA) using svd function
xsvd = svd( sweep(x30, 2, colMeans(x30), `-`) )
## (in this case, we had already swept out the column means,
##  but in case we hadn't I left it in here to demonstrate
##  that it is a critical step in PCA;
##  NOTE: R function prcomp does both sweep and svd steps,
##        though it returns results in different format)

## draw PC1 line
pc1 = xsvd$v[ , 1]
segments3d(10*c(-pc1[1], pc1[1]),
           10*c(-pc1[2], pc1[2]),
           10*c(-pc1[3], pc1[3]),
           col = 'darkgoldenrod')
text3d(11*pc1, texts='PC1', col='darkgoldenrod')

## draw PC2 line
pc2 = xsvd$v[ , 2]
segments3d(10*c(-pc2[1], pc2[1]),
           10*c(-pc2[2], pc2[2]),
           10*c(-pc2[3], pc2[3]),
           col = 'darkgoldenrod')
text3d(11*pc2, texts='PC2', col='darkgoldenrod')


## =============================================================================
## standard 2d PCA plot using ggpca function from maclearn_util.R:
ggpca(x30, tissueCoarse, cname='gene',
      cshow=3, clab=TRUE, rlab=FALSE, labrepel=TRUE,
      calpha=0.75, ralpha=0.75,
      colscale=c(gene = 'darkgray',
                 circulatory = 'firebrick3',
                 'digestive/excretory' = 'goldenrod',
                 lymphatic = 'lightseagreen',
                 nervous = 'darkorchid2',
                 other = 'darkslategray',
                 respiratory = 'dodgerblue'))


## =============================================================================
## PCA plot using all genes
ggo = ggpca(data.frame(t(x)), tissueCoarse, cname='gene',
            clab=FALSE, rlab=TRUE, labrepel=TRUE,
            calpha=0.25, ralpha=1,
            colscale=c(gene = 'darkgray',
                       circulatory = 'firebrick3',
                       'digestive/excretory' = 'goldenrod',
                       lymphatic = 'lightseagreen',
                       nervous = 'darkorchid2',
                       other = 'darkslategray',
                       respiratory = 'dodgerblue'))
