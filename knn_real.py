from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as gg
import sklearn.model_selection as model_selection
from sklearn.model_selection import ShuffleSplit
import sklearn.feature_selection as feature_selection
import sklearn.neighbors as neighbors
import sklearn.pipeline as pipeline

import warnings
warnings.filterwarnings("ignore")

import pcaextractor
from maclearn_util import safeFactorize

plt.ion()

import restricted_data
xs = restricted_data.xs
xnorms = restricted_data.xnorms
annots = restricted_data.annots
ys = restricted_data.ys
ynums = restricted_data.ynums


def pandaize(f):
    def pandaized(estimator, X, y, **kwargs):
        return f(estimator, np.array(X), safeFactorize(y), **kwargs)
    return pandaized

@pandaize
def cross_val_score_pd(estimator, X, y, **kwargs):
    return model_selection.cross_val_score(
            estimator, X, y, **kwargs)


knnClass = neighbors.KNeighborsClassifier(n_neighbors=3)
cvSchedules = {k : ShuffleSplit(n_splits = 5,
                                test_size = 0.2,
                                random_state = 123)
               for k in xnorms}
knnCvAccs = {k : np.mean(cross_val_score_pd(estimator = knnClass,
                                            X = xnorms[k],
                                            y = ys[k],
                                            cv = cvSchedules[k].split(xnorms[k])))
             for k in xnorms}


## -----------------------------------------------------------------------------
## try with univariate filter feature selection
## -----------------------------------------------------------------------------
fsKnnFitter = pipeline.Pipeline([
    ('featsel', feature_selection.SelectKBest(
            feature_selection.f_regression, k=10)),
    ('classifier', neighbors.KNeighborsClassifier(
            n_neighbors=3))
])

fsKnnCvAccs = {k : np.mean(cross_val_score_pd(estimator = fsKnnFitter,
                                              X = xnorms[k],
                                              y = ys[k],
                                              cv = cvSchedules[k].split(xnorms[k])))
               for k in xnorms}


## -----------------------------------------------------------------------------
## vary number of features used
## -----------------------------------------------------------------------------
nFeatures = [1, 2, 5, 10, 20, 50, 100, 200, 500,
             1000, 2000, 5000, 10000]
def fitKnnWithNFeat(n, setname, cv=None):
    if cv is None:
        cv = cvSchedules[setname]
    if n > xnorms[setname].shape[1]:
        return None
    fsKnnFitter = pipeline.Pipeline([
        ('featsel', feature_selection.SelectKBest(
                feature_selection.f_regression, k=n)),
        ('classifier', neighbors.KNeighborsClassifier(
                n_neighbors=3))
    ])
    return np.mean(cross_val_score_pd(estimator = fsKnnFitter,
                                      X = xnorms[setname],
                                      y = ys[setname],
                                      cv = cv.split(xnorms[setname])))

accsByNFeats = OrderedDict([(s, OrderedDict([(n, fitKnnWithNFeat(n, s))
                                             for n in nFeatures]))
                            for s in xnorms])

plotData = pd.concat([pd.DataFrame({"set" : s,
                                    "p" : p,
                                    "acc" : accsByNFeats[s][p]},
                                   index = [s + "_" + str(p)])
                      for s in accsByNFeats
                      for p in accsByNFeats[s]],
                     axis = 0)
plotData['acc'] = plotData['acc'].astype(float)

plt.close()
ggo = gg.ggplot(plotData, gg.aes(x='p', y='acc', color='set'))
ggo += gg.geom_line()
ggo += gg.scale_x_log10()
ggo += gg.theme_bw()
print(ggo)
