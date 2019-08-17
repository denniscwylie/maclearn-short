library(caret)
library(class)
library(ggplot2)

source("modelpipe.R")

load("prepared_datasets.RData")


fitModelWithNFeat = function(fitter, n, setname,
        fold=5, seed=123) {
    if (n > ncol(xnorms[[setname]])) {
        return(NA)
    }
    fsFitter = SolderedPipeFitter(
        FastTSelector(nFeat=n),
        fitter
    )
    fit = train(
        fsFitter,
        xnorms[[setname]],
        ys[[setname]],
        trControl = trainControl(
            method = "cv",
            number = fold,
            seeds = as.list(rep(seed, times=fold+1))
        )
    )
    return(fit$results$Accuracy)
}

xnames = names(xnorms)
names(xnames) = xnames

accPlot = function(accsByNFeats) {
    ggdata = data.frame(acc=accsByNFeats, row.names=names(accsByNFeats))
    ggdata$set = factor(gsub("\\..*", "",  names(accsByNFeats)),
                        levels=names(xnorms))
    ggdata$p = as.integer(gsub(".*\\.", "", names(accsByNFeats)))
    ggdata$set = factor(as.character(ggdata$set), levels=names(xnorms))
    ggobj = ggplot(data=ggdata, mapping=aes(x=p, y=acc, color=set))
    ggobj = ggobj + geom_point()
    ggobj = ggobj + geom_line(alpha=0.5)
    ggobj = ggobj + scale_x_log10(breaks=c(10, 100, 1000, 10000))
    ggobj = ggobj + theme_classic()
    ggobj = ggobj + scale_color_manual(
            values=c("darkgray", "black", "red", "dodgerblue3"))
    ggobj = ggobj + ylab("Accuracy (5-fold CV)")
    print(ggobj)
    invisible(list(data=ggdata, plot=ggobj))
}

nFeatures = c(2, 5, 10, 20, 50, 100, 200, 500,
        1000, 2000, 5000, 10000)
names(nFeatures) = as.character(nFeatures)


## -----------------------------------------------------------------
## no (err...very little) regularization
## -----------------------------------------------------------------
fitLogisticWithNFeat = function(...) {
    fitModelWithNFeat(fitter=GlmFitter(alpha=0, lambda=1e-10), ...)
}

accsByNFeats = lapply(
    X = xnames,
    FUN = function(s) {
        lapply(nFeatures, fitLogisticWithNFeat, setname=s)
    }
)
accsByNFeats = unlist(accsByNFeats)

logAccResults = accPlot(accsByNFeats)


## -----------------------------------------------------------------
## L2 regularization
## -----------------------------------------------------------------
fitL2LogisticWithNFeat = function(...) {
    fitModelWithNFeat(fitter=GlmFitter(alpha=0, lambda=NULL), ...)
}

accsByNFeatsL2 = lapply(
    X = xnames,
    FUN = function(s) {
        lapply(nFeatures, fitL2LogisticWithNFeat, setname=s)
    }
)
accsByNFeatsL2 = unlist(accsByNFeatsL2)

l2AccResults = accPlot(accsByNFeatsL2)


## -----------------------------------------------------------------
## L1 regularization
## -----------------------------------------------------------------
fitL1LogisticWithNFeat = function(...) {
    fitModelWithNFeat(fitter=GlmFitter(alpha=1, lambda=NULL), ...)
}

accsByNFeatsL1 = lapply(
    X = xnames,
    FUN = function(s) {
        lapply(nFeatures, fitL1LogisticWithNFeat, setname=s)
    }
)
accsByNFeatsL1 = unlist(accsByNFeatsL1)

l1AccResults = accPlot(accsByNFeatsL1)



## -----------------------------------------------------------------
allAccResults = rbind(
    data.frame(logAccResults$data, regularization='none'),
    data.frame(l1AccResults$data, regularization='L1'),
    data.frame(l2AccResults$data, regularization='L2')
)

ggo = ggplot(allAccResults, aes(x=p, y=acc,
                                color=set, linetype=regularization))
ggo = ggo + facet_wrap(~ set, scales='free_y')
ggo = ggo + geom_line()
ggo = ggo + theme_bw()
ggo = ggo + theme(panel.grid.minor=element_blank(),
                  panel.grid.major=element_blank())
ggo = ggo + scale_x_log10()
ggo = ggo + scale_color_manual(values=c("darkgray", "black",
                                        "red", "dodgerblue3"))
print(ggo)
