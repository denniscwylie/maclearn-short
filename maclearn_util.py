import numpy as np
import pandas as pd
import plotnine as gg

def safeFactorize(series):
    if "factorize" in dir(series):
        return series.factorize()[0]
    else:
        uniqSer = series.unique()
        out = pd.Series(np.zeros(len(series)))
        out.index = series.index
        for i in range(1, len(uniqSer)):
            out.loc[series == uniqSer[i]] = i
        return out

def svdForPca(x, center="col", scale="none", pandaize=True):
    if min(x.std(axis=0)) == 0:
        return None
    xhere = x.copy()
    if center in ['row', 'both']:
        xRowAvs = xhere.mean(axis=1)
        xhere = xhere.add(-xRowAvs, axis=0)
    if center in ['col', 'both']:
        xColAvs = xhere.mean(axis=0)
        xhere = xhere.add(-xColAvs, axis=1)
    if scale == 'row':
        rowSds = xhere.std(axis=1)
        xhere = xhere.divide(rowSds, axis=0)
    elif scale == 'col':
        colSds = xhere.std(axis=0)
        xhere = xhere.divide(colSds, axis=1)
    xsvd = np.linalg.svd(xhere, full_matrices=False)
    if pandaize:
        xsvd = (
            pd.DataFrame(xsvd[0], index=x.index),
            pd.Series(xsvd[1]),
            pd.DataFrame(xsvd[2], columns=x.columns)
        )
    return xsvd

def ggpca(x, y=None, center='col', scale='none',
          rlab=False, clab=False, cshow=None,
          rsize=4, csize=2, lsize=10, lnudge=0.03,
          ralpha=0.6, calpha=1.0, clightalpha=0,
          rname='sample', cname='variable', lname='',
          grid=True, printit=False, xsvd=None,
          invert1=False, invert2=False, colscale=None,
          **kwargs):
    if cshow is None:
        cshow = x.shape[1]
    if rlab is not None and isinstance(rlab, bool):
        rlab = x.index if rlab else ''
    if clab is not None and isinstance(clab, bool):
        clab = x.columns if clab else ''
    if y is not None:
        pass
    x = x.loc[:, x.isnull().sum(axis=0) == 0]
    if xsvd is None:
        xsvd = svdForPca(x, center, scale)
    rsf = np.max(xsvd[0].iloc[:, 0]) - np.min(xsvd[0].iloc[:, 0])
    csf = np.max(xsvd[2].iloc[0, :]) - np.min(xsvd[2].iloc[0, :])
    sizeRange = sorted([csize, rsize])
    alphaRange = sorted([calpha, ralpha])
    ggd = pd.DataFrame({
        'PC1' : xsvd[0].iloc[:, 0] / rsf,
        'PC2' : xsvd[0].iloc[:, 1] / rsf,
        'label' : rlab,
        'size' : rsize,
        'alpha' : ralpha
    })
    cclass = []
    if cshow > 0:
        cdata = pd.DataFrame({
            'PC1' : xsvd[2].iloc[0, :] / csf,
            'PC2' : xsvd[2].iloc[1, :] / csf,
            'label' : clab,
            'size' : csize,
            'alpha' : calpha
        })
        if cshow < x.shape[1]:
            cscores = cdata['PC1']**2 + cdata['PC2']**2
            keep = cscores.sort_values(ascending=False).head(cshow).index
            if clightalpha > 0:
                cdata.loc[~cdata.index.isin(keep), 'label'] = ''
                cdata.loc[~cdata.index.isin(keep), 'alpha'] = clightalpha
                alphaRange = [np.min([alphaRange[0], clightalpha]),
                              np.max([alphaRange[1], clightalpha])]
            else:
                cdata = cdata.loc[cdata.index.isin(keep)]
        ggd = pd.concat([cdata, ggd])
        cclass = [cname] * cdata.shape[0]
    if invert1:
        ggd['PC1'] = -ggd['PC1']
    if invert2:
        ggd['PC2'] = -ggd['PC2']
    if y is not None:
        ggd['class'] = cclass + list(y.loc[x.index])
    else:
        ggd['class'] = cclass + ([rname] *  x.shape[0])
    ggo = gg.ggplot(ggd, gg.aes(
        x = 'PC1',
        y = 'PC2',
        color = 'class',
        size = 'size',
        alpha = 'alpha',
        label = 'label'
    ))
    ggo += gg.geom_hline(yintercept=0, color='gray')
    ggo += gg.geom_vline(xintercept=0, color='gray')
    ggo += gg.geom_point()
    ggo += gg.theme_bw()
    ggo += gg.geom_text(nudge_y=lnudge, size=lsize, show_legend=False)
    if colscale is None and len(ggd['class'].unique()) < 8:
        colscale = ['darkslategray', 'goldenrod', 'lightseagreen',
                    'orangered', 'dodgerblue', 'darkorchid']
        colscale = colscale[0:(len(ggd['class'].unique())-1)] + ['gray']
        if len(colscale) == 2 and cshow > 0:
            colscale = ['black', 'darkgray']
        if len(colscale) == 2 and cshow == 0:
            colscale = ['black', 'red']
        if len(colscale) == 3:
            colscale = ['black', 'red', 'darkgray']
    ggo += gg.scale_color_manual(values=colscale, name=lname)
    ggo += gg.scale_size_continuous(guide=False, range=sizeRange)
    ggo += gg.scale_alpha_continuous(guide=False, range=alphaRange)
    ggo += gg.xlab('PC1 (' +
                   str(np.round(100*xsvd[1][0]**2 / ((xsvd[1]**2).sum()), 1)) +
                   '% explained var.)')
    ggo += gg.ylab('PC2 (' +
                   str(np.round(100*xsvd[1][1]**2 / ((xsvd[1]**2).sum()), 1)) +
                   '% explained var.)')
    if not grid:
        ggo += gg.theme(panel_grid_minor = gg.element_blank(),
                        panel_grid_major = gg.element_blank(),
                        panel_background = gg.element_blank())
    ggo += gg.theme(axis_ticks = gg.element_blank(),
                    axis_text_x = gg.element_blank(),
                    axis_text_y = gg.element_blank())
    if printit:
        print(ggo)
    return ggo
