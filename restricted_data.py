import numpy as np
import pandas as pd

import maclearn_util

import load_data
xs = load_data.xs
annots = load_data.annots

def meanCenterAndImpute(x, axis=0, imputeAt=None):
    if imputeAt is None:
        imputeAt = np.ceil(x.max().max())
    geneHasNans = (np.isnan(x).sum(axis=axis) > 0)
    if axis == 0:
        xnonans = x[ x.columns[~geneHasNans] ]
    elif axis == 1:
        xnonans = x.loc[~geneHasNans]
    means = xnonans.mean(axis=1-axis)
    out = x.copy()
    out[np.isnan(out)] = imputeAt
    return out.add(-means, axis=axis)

xnorms = {}
## shen set already normalized
xnorms['shen'] = xs['shen'].copy()
## patel set already normalized
xnorms['patel'] = xs['patel'].copy()
## simple normalization of montastier data set
xnorms['montastier'] = meanCenterAndImpute(xs['montastier'])
## hess set already normalized
xnorms['hess'] = xs['hess'].copy()

patelSubtype = annots['patel'].SubType
patelKeepers = ((patelSubtype == 'subtype: Mes') |
                (patelSubtype == 'subtype: Pro'))
patelKeepers = annots['patel'].index[patelKeepers]

xs['patel'] = xs['patel'].loc[patelKeepers]
xnorms['patel'] = xnorms['patel'].loc[patelKeepers]
annots['patel'] = annots['patel'].loc[patelKeepers]


montastierTime = annots['montastier'].Time
montastierKeepers = ((montastierTime == 'C1') |
                     (montastierTime == 'C2'))

xs['montastier'] = xs['montastier'].loc[montastierKeepers]
xnorms['montastier'] = xnorms['montastier'].loc[montastierKeepers]
annots['montastier'] = annots['montastier'].loc[montastierKeepers]


## -----------------------------------------------------------------
## extract ys
## -----------------------------------------------------------------
ys = {
    'shen' : annots['shen'].Nervous,
    'patel' : annots['patel'].SubType,
    'montastier' : annots['montastier'].Time,
    'hess' : annots['hess'].pCRtxt
}

ynums = {k : maclearn_util.safeFactorize(ys[k]) for k in ys}
