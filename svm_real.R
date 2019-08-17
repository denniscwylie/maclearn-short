library(caret)
library(e1071)
library(ggplot2)

source("maclearn_util.R")
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
    return(fit)
}

xnames = names(xnorms)
names(xnames) = xnames

fsSvmLinModels = lapply(xnames, fitModelWithNFeat,
                        fitter=SvmFitter(kernel="linear", cost=1), n=10)
fsSvmLinAccs = sapply(fsSvmLinModels, function(u) {u$results$Accuracy})

fsSvmRadModels = lapply(xnames, fitModelWithNFeat,
                        fitter=SvmFitter(kernel="radial", cost=1), n=10)
fsSvmRadAccs = sapply(fsSvmRadModels, function(u) {u$results$Accuracy})
