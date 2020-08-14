from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as gg
import sklearn.model_selection as model_selection
from sklearn.model_selection import ShuffleSplit
import sklearn.feature_selection as feature_selection
import sklearn.linear_model as linear_model
import sklearn.pipeline as pipeline

import warnings
warnings.filterwarnings("ignore")

plt.ion()

import restricted_data
xs = restricted_data.xs
xnorms = restricted_data.xnorms
annots = restricted_data.annots
ys = restricted_data.ys
ynums = restricted_data.ynums

cvSchedules = {k : ShuffleSplit(n_splits = 5,
                                test_size = 0.2,
                                random_state = 123)
               for k in xnorms}


def pandaize(f):
    def pandaized(estimator, X, y, **kwargs):
        return f(estimator, np.array(X), y, **kwargs)
    return pandaized

@pandaize
def cross_val_score_pd(estimator, X, y, **kwargs):
    return model_selection.cross_val_score(
            estimator, X, y, **kwargs)

def fitModelWithNFeat(fitter, n, setname, cv=None):
    if cv is None:
        cv = cvSchedules[setname]
    if n > xnorms[setname].shape[1]:
        return None
    fsFitter = pipeline.Pipeline([
        ('featsel', feature_selection.SelectKBest(
                feature_selection.f_regression, k=n)),
        ('classifier', fitter)
    ])
    return np.mean(cross_val_score_pd(estimator = fsFitter,
                                      X = xnorms[setname],
                                      y = ynums[setname],
                                      cv = cv.split(xnorms[setname])))

def accPlot(accsByNFeats):
    plotdata = []
    for s in accsByNFeats:
        plotdata.append(pd.concat([pd.DataFrame({"p" : p,
                                                 "acc" : accsByNFeats[s][p],
                                                 "set" : s},
                                                index = [str(p)])
                                   for p in accsByNFeats[s]],
                                  axis = 0))
    ggd = pd.concat(plotdata)
    ggd['acc'] = ggd['acc'].astype(float)
    ggo = gg.ggplot(ggd, gg.aes(x='p', y='acc', color='set'))
    ggo += gg.geom_line(alpha=0.5)
    ggo += gg.geom_point()
    ggo += gg.theme_bw()
    ggo += gg.scale_x_log10(breaks=[10, 100, 1000, 10000])
    ggo += gg.scale_color_manual(values=['darkgray', 'black',
                                         'red', 'dodgerblue'])
    ggo += gg.ylab('Accuracy (5-fold CV)')
    print(ggo)
    return ggd

nFeatures = [2, 5, 10, 20, 50, 100, 200, 500,
             1000, 2000, 5000, 10000]


## -----------------------------------------------------------------
## no (err...very little) regularization
## -----------------------------------------------------------------
def fitLogisticWithNFeat(**kwargs):
    fitter = linear_model.LogisticRegression(penalty="l2", C=1e10)
    return fitModelWithNFeat(fitter=fitter, **kwargs)

nFeatNoReg = [2, 5, 10, 20, 50, 100, 200]
accsByNFeats = OrderedDict([(s,
                             OrderedDict([(
                                 n,
                                 fitLogisticWithNFeat(n=n, setname=s))
                                          for n in nFeatNoReg]))
                            for s in xnorms])
for s in accsByNFeats:
    for n in accsByNFeats[s]:
        if n > xnorms[s].shape[0]:
            accsByNFeats[s][n] = None

plt.close()
logAccResults = accPlot(accsByNFeats)


## -----------------------------------------------------------------
## L2 regularization
## -----------------------------------------------------------------
def fitL2LogisticWithNFeat(**kwargs):
    fitter = linear_model.LogisticRegression(penalty="l2", C=1)
    return fitModelWithNFeat(fitter=fitter, **kwargs)

accsByNFeatsL2 = OrderedDict([(s,
                               OrderedDict([(
                                   n,
                                   fitL2LogisticWithNFeat(n=n, setname=s))
                                            for n in nFeatures]))
                              for s in xnorms])

plt.close()
l2AccResults = accPlot(accsByNFeatsL2)


## -----------------------------------------------------------------
## L1 regularization
## -----------------------------------------------------------------
def fitL1LogisticWithNFeat(**kwargs):
    fitter = linear_model.LogisticRegression(penalty="l1", C=1, solver="saga")
    return fitModelWithNFeat(fitter=fitter, **kwargs)

accsByNFeatsL1 = OrderedDict([(s,
                               OrderedDict([(
                                   n,
                                   fitL1LogisticWithNFeat(n=n, setname=s))
                                            for n in nFeatures]))
                              for s in xnorms])

plt.close()
l1AccResults = accPlot(accsByNFeatsL1)



## -----------------------------------------------------------------
logAccResults['regularization'] = '-'
l1AccResults['regularization'] = 'L1'
l2AccResults['regularization'] = 'L2'
allAccResults = pd.concat([logAccResults, l1AccResults, l2AccResults])

plt.close()
ggo = gg.ggplot(allAccResults, gg.aes(x='p', y='acc',
                                      color='set',
                                      linetype='regularization'))
ggo += gg.facet_wrap('~ set', scales='free_y')
ggo += gg.geom_line()
ggo += gg.theme_bw()
ggo += gg.theme(panel_grid_minor=gg.element_blank(),
                panel_grid_major=gg.element_blank())
ggo += gg.scale_x_log10()
ggo += gg.scale_color_manual(values=['darkgray', 'black',
                                     'red', 'dodgerblue'])
print(ggo)
