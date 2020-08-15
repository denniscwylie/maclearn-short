rowSds = function(x, na.rm=FALSE) {
    n = ncol(x)
    return(sqrt((n/(n-1)) * (
            rowMeans(x*x, na.rm=na.rm) - rowMeans(x, na.rm=na.rm))))
}
colSds = function(x, na.rm=FALSE) {
    n = nrow(x)
    return(sqrt((n/(n-1)) * (
            colMeans(x*x, na.rm=na.rm) - colMeans(x, na.rm=na.rm))))
}


XTransformer = function(f) {
    f = f
    learner = function(x, y, ...) {list(transform=f)}
    class(learner) = c("XTransformer", "ModelPipe", class(learner))
    return(learner)
}

PValueSelector = function(f=NULL, threshold=0.05, fdr="fdr", 
        nFeat=NULL, vectorize=TRUE) {
    f = f
    threshold = threshold
    fdr = fdr
    nFeat = nFeat
    vectorize = vectorize
    if (length(f) == 0) {
        vectorize = FALSE
        f = function(x, y, ...) {
            yFeat = y
            if (is.factor(yFeat)) {
                yFeat = as.integer(yFeat)
            }
            apply(X=x, MARGIN=2, FUN=function(z) {
                t.test(z[yFeat==1], z[yFeat==2], ...)[["p.value"]]
            })
        }
    }
    learner = function(x, y, ...) {
        featurePVals = if (vectorize) {
            apply(X=x, MARGIN=2, FUN=function(z) {
                f(z, y, ...)
            })
        } else {
            f(x, y, ...)
        }
        selectedFeatures = if (length(nFeat) > 0) {
            names(sort(featurePVals))[1:nFeat]
        } else {
            if (length(fdr) > 0) {
                featurePVals = p.adjust(featurePVals, method=fdr)
            }
            names(featurePVals[featurePVals < threshold])
        }
        return(list(
            selectedFeatures = selectedFeatures,
            p = featurePVals,
            transform = function(x, ...) {
                x[ , selectedFeatures, drop=FALSE]
            }
        ))
    }
    class(learner) = c("PValueSelector", "ModelPipe", class(learner))
    return(learner)
}

ModelFitter = function(f,
        predictor=predict, predictionProcessor=identity) {
    predictor = predictor
    predictionProcessor = predictionProcessor
    learner = function(x, y, ...) {
        fit = f(x, y, ...)
        fitOut = list(
            fit = fit,
            predict = function(x, ...) {
                predictionProcessor(predictor(fit, x, ...))
            }
        )
        class(fitOut) = "ModelFit"
        return(fitOut)
    }
    class(learner) = c("ModelFitter", "ModelPipe", class(learner))
    return(learner)
}

SolderedPipeFitter = function(...) {
    piping = list(...)
    learner = function(x, y, ...) {
        fit = list()
        for (i in 1:length(piping)) {
            fit[[i]] = (piping[[i]])(x, y, ...)
            if (is(fit[[i]], "SolderedPipe")) {
                x = transform(fit[[i]], x, ...)
            } else if ("transform" %in% names(fit[[i]])) {
                x = fit[[i]]$transform(x, y, ...)
            }
        }
        class(fit) = c("SolderedPipe", "ModelPipe")
        return(fit)
    }
    class(learner) = c("SolderedPipeFitter", "ModelFitter",
            "SolderedPipe", "ModelPipe", class(learner))
    return(learner)
}
SolderedPipe = SolderedPipeFitter
solder = SolderedPipeFitter

transform.ModelPipe = function(obj, x, ...) {
    if (!is(obj, "SolderedPipe")) {
        obj = solder(obj)
    }
    for (subobj in obj) {
        if (is(subobj, "SolderedPipe")) {
            x = transform(subobj, x, ...)
        } else if ("transform" %in% names(subobj)) {
            x = subobj$transform(x, ...)
        }
    }
    return(x)
}

predict.ModelPipe = function(obj, x, ..., level=0) {
    if (level == 0) {
        x = transform(obj, x, ...)
    }
    if (!is(obj, "SolderedPipe")) {
        obj = solder(obj)
    }
    for (i in length(obj):1) {
        if (is(obj[[i]], "SolderedPipe")) {
            out = predict(obj[[i]], x, ..., level=level+1)
            if (!all(is.na(out))) {
                return(out)
            }
        } else if ("predict" %in% names(obj[[i]])) {
            return(obj[[i]]$predict(x, ...))
        }
    }
    return(NA)
    ## stop("This ModelPipe object does not support predict.")
}

predict.ModelFit = function(obj, x, ...) {obj$predict(x, ...)}


caretize = function(fitpipe, lev=NULL, threshold=0.5, ...,
        type=NULL, library=NULL, loop=NULL,
        parameters=NULL, grid=NULL) {
    lev = lev
    threshold = threshold
    if (length(type) == 0) {
        type = ifelse(length(lev)==0, "Regression", "Classification")
    }
    if (type == "Classification") {
        caretPredict = function(modelFit, newdata, ...) {
            ifelse(
                predict(modelFit, newdata) < threshold,
                lev[1],
                lev[2]
            )
        }
    } else {
        caretPredict = function(modelFit, newdata, ...) {
            predict(modelFit, newdata)
        }
    }
    if (length(parameters) == 0) {
        parameters = data.frame(
            parameter = "ignored",
            class = "numeric",
            label = "Ignored"
        )
    }
    if (length(grid) == 0) {
        grid = function(x, y, len=NULL, ...) {data.frame(ignored=0)}
    }
    return(list(
        library = library,
        type = type,
        loop = loop,
        parameters = parameters,
        grid = grid,
        fit = fitpipe,
        predict = caretPredict,
        prob = function(modelFit, newdata, ...) {
            preds = predict(modelFit, newdata)
            out = data.frame(
                lev1 = 1 - preds,
                lev2 = preds
            )
            colnames(out) = lev
            return(out)
        }
    ))
}


train.ModelFitter = function(
        fitpipe,
        x,
        y,
        threshold = 0.5,
        ...,
        method = "repeatedcv",
        number = 10,
        repeats = 1,
        trControl = trainControl(method, number, repeats),
        tuneGrid = NULL,
        parameters = NULL,
        grid = NULL,
        type = NULL,
        library = NULL,
        loop = NULL) {
    caretizedPipe = caretize(
        fitpipe,
        lev = levels(y),
        threshold = threshold,
        type = type,
        library = library,
        loop = loop,
        parameters = parameters,
        grid = grid
    )
    return(train(
        x = x,
        y = y,
        method = caretizedPipe,
        trControl = trControl,
        tuneGrid = tuneGrid,
        ...
    ))
}


FastTSelector = function(threshold=0.05, fdr="fdr", nFeat=NULL) {
    threshold = threshold
    fdr = fdr
    nFeat = nFeat
    require(genefilter)
    selector = PValueSelector(
        f = function(x, y, ...) {colttests(as.matrix(x), y)$p.value},
        threshold = threshold,
        fdr = fdr,
        nFeat = nFeat
    )
    class(selector) = c("FastTSelector", class(selector))
    return(selector)
}


KnnFitter = function(k=5) {
    k = k
    out = ModelFitter(
        f = function(x, y, ...) {
            list(k=k, x=x, y=y)
        },
        predictor = function(obj, x, ...) {
            require(class)
            knnObj = knn(train=obj$x, test=x, cl=obj$y, k=obj$k,
                    prob=TRUE, use.all=TRUE)
            predictions = sign(as.numeric(knnObj) - 1.5)
            predictions = predictions * (attr(knnObj, "prob") - 0.5)
            predictions = predictions + 0.5
            names(predictions) = rownames(x)
            return(predictions)
        }
    )
    class(out) = c("KnnFitter", class(out))
    return(out)
}


GlmFitter = function(fam="binomial", alpha=0, lambda=NULL) {
    fam = fam
    al = alpha
    lam = lambda
    out = ModelFitter(
        f = function(x, y, ..., lambda=lam) {
            require(glmnet)
            if (length(lambda) == 0) {
                cvOut = cv.glmnet(as.matrix(x), y,
                        family=fam, alpha=al)
                lambda = cvOut$lambda.min
            }
            out = glmnet(as.matrix(x), y,
                    family=fam, alpha=al, lambda=lam)
            out$lambda.min = lambda
            return(out)
        },
        predictor = function(obj, x, ..., lambda) {
            if (missing(lambda)) {
                lambda = obj$lambda.min
            }
            return(predict(obj, as.matrix(x),
                    s=lambda, type="response")[ , 1])
        }
    )
    class(out) = c("GlmFitter", class(out))
    return(out)
}


SvmFitter = function(...) {
    svmArgs = list(...)
    svmArgs$probability = TRUE
    out = ModelFitter(
        f = function(x, y, ...) {
            require(e1071)
            if (!all(unique(as.character(y)) == levels(y))) {
                ord = order(y)
                x = x[ord, , drop=FALSE]
                y = y[ord]
            }
            do.call(svm, c(list(x=x, y=y), svmArgs))
        },
        predictor = function(obj, x, ...) {
            predictions = attr(predict(obj, x,
                    probability=TRUE, ...), "probabilities")[ , 2]
            names(predictions) = rownames(x)
            return(predictions)
        }
    )
    class(out) = c("SvmFitter", class(out))
    return(out)
}
