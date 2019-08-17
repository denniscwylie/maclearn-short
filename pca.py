#!/usr/bin/env python3

import numpy as np
import pandas as pd

from maclearn_util import ggpca

x = pd.read_csv('rnaseq/shen2012/19-tissues-expr.tsv.gz',
                sep='\t', header=0, index_col=0)

## extract tissue types from colnames of x
tissue = pd.Series(
    x.columns.str.replace('\d*(-.*)?', ''),
    index = x.columns
)
tissueCoarse = pd.Series({
    'boneMarrow' : 'lymphatic',
    'brain' : 'nervous',
    'cerebellum' : 'nervous',
    'cortex' : 'nervous',
    'heart' : 'circulatory',
    'intestine' : 'digestive/excretory',
    'kidney' : 'digestive/excretory',
    'limb' : 'other',
    'liver' : 'digestive/excretory',
    'lung' : 'respiratory',
    'mef' : 'other',
    'mESC' : 'other',
    'olfactory' : 'nervous',
    'placenta' : 'other',
    'spleen' : 'lymphatic',
    'testes' : 'other',
    'thymus' : 'lymphatic'
}).reindex(tissue.values)
tissueCoarse.index = tissue.index


## =============================================================================
## PCA plot using all genes
ggpca(x.T, tissueCoarse, cname='gene',
      cshow=False, rlab=True,
      calpha=0.25, ralpha=1,
      colscale=['firebrick', 'goldenrod', 'lightseagreen',
                'darkorchid', 'darkslategray', 'dodgerblue'])

