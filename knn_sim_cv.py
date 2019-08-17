from collections import OrderedDict
import itertools
import numpy as np
import pandas as pd
import plotnine as gg
import sklearn
from sklearn.neighbors import KNeighborsClassifier
import sklearn.model_selection
from sklearn.model_selection import cross_val_score

import sim_data


x2_train = sim_data.simulate2Group(n = 100,
                                   p = 2,
                                   effect = [1.25] * 2)
knnClass = KNeighborsClassifier(n_neighbors=3)
cvAccs = cross_val_score(estimator = knnClass,
                         X = np.array(x2_train['x']),
                         y = np.array(x2_train['y']),
                         cv = 5)
cvAccEst = np.mean(cvAccs)
knnClass.fit(np.array(x2_train['x']), np.array(x2_train['y']))

x2_test = sim_data.simulate2Group(n = 100,
                                  p = 2,
                                  effect = [1.25] * 2)
knnTest = pd.Series(knnClass.predict(x2_test['x']),
                    index = x2_test['y'].index)
testAccEst = (np.sum(np.diag(pd.crosstab(knnTest, x2_test['y']))) /
              (1.0 * np.sum(np.sum(pd.crosstab(knnTest, x2_test['y'])))))

def expandGrid(od):
    cartProd = list(itertools.product(*od.values()))
    return pd.DataFrame(cartProd, columns=od.keys())

parVals = OrderedDict()
parVals['n'] = [100]
parVals['p'] = [2, 5, 10, 25, 100, 500]
parVals['k'] = [3, 5, 10, 25]
parGrid = expandGrid(parVals)
parGrid['effect'] = 2.5
parGrid['effect'] = parGrid['effect'] / np.sqrt(parGrid['p'])


def knnSimulate(param, nFold=5):
    trainSet = sim_data.simulate2Group(
        n = int(param['n']),
        p = int(param['p']),
        effect = [param['effect']] * int(param['p'])
    )
    knnClass = KNeighborsClassifier(n_neighbors=int(param['k']))
    cvAccs = cross_val_score(estimator = knnClass,
                             X = np.array(trainSet['x']),
                             y = np.array(trainSet['y']),
                             cv = nFold)
    knnClass.fit(np.array(trainSet['x']), np.array(trainSet['y']))
    testSet = sim_data.simulate2Group(
        n = int(param['n']),
        p = int(param['p']),
        effect = [param['effect']] * int(param['p'])
    )
    out = OrderedDict()
    out['p'] = param['p']
    out['k'] = param['k']
    out['train'] = trainSet
    out['test'] = testSet
    out['testPreds'] = knnClass.predict(testSet['x'])
    out['testProbs'] = knnClass.predict_proba(testSet['x'])
    out['cvAccuracy'] = np.mean(cvAccs)
    out['testTable'] = pd.crosstab(
        pd.Series(out['testPreds'], index=testSet['y'].index),
        testSet['y']
    )
    out['testAccuracy'] = (np.sum(np.diag(out['testTable'])) /
                           (1.0 * np.sum(np.sum(out['testTable']))))
    return out


repeatedKnnResults = []
for r in range(5):
    repeatedKnnResults.extend(knnSimulate(parGrid.iloc[i])
                              for i in range(parGrid.shape[0]))

knnResultsSimplified = pd.DataFrame([(x['p'],
                                      x['k'],
                                      x['cvAccuracy'],
                                      x['testAccuracy'])
                                     for x in repeatedKnnResults],
                                    columns = ['p',
                                               'k',
                                               'cvAccuracy',
                                               'testAccuracy'])


ggdata = pd.concat(
    [pd.DataFrame({'p' : knnResultsSimplified.p,
                   'k' : knnResultsSimplified.k.apply(int),
                   'type' : 'cv',
                   'Accuracy' : knnResultsSimplified.cvAccuracy}),
     pd.DataFrame({'p' : knnResultsSimplified.p,
                   'k' : knnResultsSimplified.k.apply(int),
                   'type' : 'test',
                   'Accuracy' : knnResultsSimplified.testAccuracy})],
    axis = 0
)

ggo = gg.ggplot(ggdata, gg.aes(x='p', y='Accuracy',
                               color='type', group='type', linetype='type'))
ggo += gg.scale_x_log10()
ggo += gg.geom_point(alpha=0.6)
ggo += gg.stat_smooth()
ggo += gg.facet_wrap('~ k')
ggo += gg.theme_bw()
print(ggo)
