# Results

Scikit-learn [docs](https://scikit-learn.org/stable/modules/multiclass.html).

- Multiclass Classification: classification task with more than two classes. Each sample can only be labelled as one class.

- Multilabel Classification: classification task labelling each sample with `x` labels from `n_classes` possible classes, where `x` can be 0 to `n_classes` inclusive.
    - All Multiclass Classification algorithms can become multilabel classification algorithms using the [MultiOutputClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html#sklearn.multioutput.MultiOutputClassifier).

## Multilabel Algorithms

The following algorithms are Scikit learn Multilabel supporting algorithms.
Each lead is evaluates separately, then the results are stacked together and passed to a `RandomForestClassifier` ensemble.

```python
sklearn.tree.DecisionTreeClassifier
sklearn.tree.ExtraTreeClassifier
sklearn.ensemble.ExtraTreesClassifier
sklearn.neighbors.KNeighborsClassifier
sklearn.neural_network.MLPClassifier
sklearn.neighbors.RadiusNeighborsClassifier
sklearn.ensemble.RandomForestClassifier
sklearn.linear_model.RidgeClassifierCV # added MultiOutputClassification, even though supports Fit on n-label y
```

### No feature selection

Each of the 12 leads evaluated separately using lead classifier, then label output probabilities are stacked and passed as input into a RandomForestClassifier.

| Lead Classifier | Dataset | Accuracy | F_Measure | F_Beta | G_Beta | AUROC | AUPRC |
|-----------------|---------|----------|-----------|--------|--------|-------|-------|
| DecisionTreeClassifier | cv5-0/Validation | 0.9438 | 0.6143 | 0.5340 | 0.3450 | 0.8049 | 0.6455 |
| DecisionTreeClassifier | cv5-1/Validation | 0.9448 | 0.6244 | 0.5504 | 0.3660 | 0.8059 | 0.6552 |
| DecisionTreeClassifier | cv5-2/Validation | 0.9458 | 0.6214 | 0.5393 | 0.3429 | 0.8374 | 0.6821 |
| DecisionTreeClassifier | cv5-3/Validation | 0.9428 | 0.5910 | 0.5151 | 0.3323 | 0.8095 | 0.6388 |
| DecisionTreeClassifier | cv5-4/Validation | 0.9435 | 0.6118 | 0.5291 | 0.3360 | 0.8233 | 0.6730 |
| ExtraTreeClassifier | cv5-0/Validation | 0.9312 | 0.5119 | 0.4272 | 0.2575 | 0.8095 | 0.6233 |
| ExtraTreeClassifier | cv5-1/Validation | 0.9285 | 0.4716 | 0.3913 | 0.2313 | 0.8090 | 0.6085 |
| ExtraTreeClassifier | cv5-2/Validation | 0.9300 | 0.4673 | 0.3858 | 0.2283 | 0.8495 | 0.6471 |
| ExtraTreeClassifier | cv5-3/Validation | 0.9276 | 0.4650 | 0.3839 | 0.2255 | 0.7962 | 0.5850 |
| ExtraTreeClassifier | cv5-4/Validation | 0.9270 | 0.4590 | 0.3717 | 0.2124 | 0.8193 | 0.6308 |
| ExtraTreesClassifier | cv5-0/Validation | 0.9387 | 0.5652 | 0.4841 | 0.3214 | 0.7168 | 0.5702 |
| ExtraTreesClassifier | cv5-1/Validation | 0.9334 | 0.5070 | 0.4271 | 0.2664 | 0.7591 | 0.5745 |
| ExtraTreesClassifier | cv5-2/Validation | 0.9392 | 0.5535 | 0.4679 | 0.2950 | 0.7823 | 0.6052 |
| ExtraTreesClassifier | cv5-3/Validation | 0.9325 | 0.5080 | 0.4273 | 0.2633 | 0.7320 | 0.5566 |
| ExtraTreesClassifier | cv5-4/Validation | 0.9357 | 0.5378 | 0.4532 | 0.2867 | 0.7538 | 0.5988 |
| KNeighborsClassifier | cv5-0/Validation | 0.9111 | 0.3786 | 0.3062 | 0.1638 | 0.8454 | 0.5727 |
| KNeighborsClassifier | cv5-1/Validation | 0.9116 | 0.3854 | 0.3181 | 0.1738 | 0.8603 | 0.5779 |
| KNeighborsClassifier | cv5-2/Validation | 0.9140 | 0.3821 | 0.3106 | 0.1677 | 0.8652 | 0.5982 |
| KNeighborsClassifier | cv5-3/Validation | 0.9116 | 0.3524 | 0.2843 | 0.1520 | 0.8496 | 0.5644 |
| KNeighborsClassifier | cv5-4/Validation | 0.9142 | 0.4038 | 0.3282 | 0.1793 | 0.8606 | 0.6012 |
| MLPClassifier | cv5-0/Validation | 0.9421 | 0.6189 | 0.5585 | 0.3672 | 0.9121 | 0.7298 |
| MLPClassifier | cv5-1/Validation | 0.9446 | 0.6318 | 0.5772 | 0.3875 | 0.9140 | 0.7359 |
| MLPClassifier | cv5-2/Validation | 0.9490 | 0.6346 | 0.5768 | 0.3870 | 0.9008 | 0.7333 |
| MLPClassifier | cv5-3/Validation | 0.9440 | 0.5948 | 0.5300 | 0.3410 | 0.8980 | 0.7072 |
| MLPClassifier | cv5-4/Validation | 0.9474 | 0.6587 | 0.5961 | 0.3969 | 0.9258 | 0.7718 |
| RandomForestClassifier | cv5-0/Validation | 0.9505 | 0.6734 | 0.6091 | 0.4310 | 0.7684 | 0.6343 |
| RandomForestClassifier | cv5-1/Validation | 0.9491 | 0.6507 | 0.5829 | 0.3911 | 0.7627 | 0.6308 |
| RandomForestClassifier | cv5-2/Validation | 0.9517 | 0.6640 | 0.5857 | 0.3895 | 0.7942 | 0.6657 |
| RandomForestClassifier | cv5-3/Validation | 0.9448 | 0.6251 | 0.5540 | 0.3679 | 0.7602 | 0.6100 |
| RandomForestClassifier | cv5-4/Validation | 0.9485 | 0.6730 | 0.5906 | 0.3931 | 0.7900 | 0.6730 |
| RidgeClassifierCV | cv5-0/Validation | 0.9492 | 0.6859 | 0.6524 | 0.4527 | 0.8347 | 0.6418 |
| RidgeClassifierCV | cv5-1/Validation | 0.9488 | 0.6750 | 0.6396 | 0.4356 | 0.8422 | 0.6608 |
| RidgeClassifierCV | cv5-2/Validation | 0.9488 | 0.6788 | 0.6343 | 0.4259 | 0.8306 | 0.6536 |
| RidgeClassifierCV | cv5-3/Validation | 0.9481 | 0.6727 | 0.6308 | 0.4229 | 0.8095 | 0.6289 |
| RidgeClassifierCV | cv5-4/Validation | 0.9497 | 0.6968 | 0.6563 | 0.4520 | 0.8506 | 0.6697 |

#### Statistical Mean of Columns

| Lead Classifier | Accuracy | F_Measure | F_Beta | G_Beta | AUROC | AUPRC |
|-----------------|----------|-----------|--------|--------|-------|-------|
| DecisionTreeClassifier | 0.9441 | 0.6126 | 0.5336 | 0.3444 | 0.8162 | 0.6589 |
| ExtraTreeClassifier | 0.9289 | 0.4750 | 0.3920 | 0.2310 | 0.8167 | 0.6189 |
| ExtraTreesClassifier | 0.9359 | 0.5343 | 0.4519 | 0.2866 | 0.7488 | 0.5811 |
| KNeighborsClassifier | 0.9125 | 0.3805 | 0.3095 | 0.1673 | 0.8562 | 0.5829 |
| MLPClassifier | 0.9454 | 0.6278 | 0.5677 | 0.3759 | 0.9101 | 0.7356 |
| RandomForestClassifier | 0.9489 | 0.6572 | 0.5845 | 0.3945 | 0.7751 | 0.6428 |
| RidgeClassifierCV | 0.9489 | 0.6818 | 0.6427 | 0.4378 | 0.8335 | 0.6510 |


### Using Random Forest Classifier as Feature Selection

Same as above, but a Scikit-learn pipeline with a RandomForestClassifier to select important features is applied on each of the twelve leads before lead classification.

| Lead Classifier | Dataset | Accuracy | F_Measure | F_Beta | G_Beta | AUROC | AUPRC |
|-----------------|---------|----------|-----------|--------|--------|-------|-------|
| DecisionTreeClassifier | cv5-0/Validation | 0.9469 | 0.6444 | 0.5731 | 0.3795 | 0.8034 | 0.6417 |
| DecisionTreeClassifier | cv5-1/Validation | 0.9448 | 0.6200 | 0.5432 | 0.3566 | 0.7973 | 0.6519 |
| DecisionTreeClassifier | cv5-2/Validation | 0.9453 | 0.6151 | 0.5367 | 0.3448 | 0.8384 | 0.6879 |
| DecisionTreeClassifier | cv5-3/Validation | 0.9438 | 0.5993 | 0.5235 | 0.3410 | 0.8051 | 0.6375 |
| DecisionTreeClassifier | cv5-4/Validation | 0.9450 | 0.6281 | 0.5465 | 0.3511 | 0.8248 | 0.6759 |
| ExtraTreeClassifier | cv5-0/Validation | 0.9343 | 0.5238 | 0.4426 | 0.2698 | 0.8131 | 0.6226 |
| ExtraTreeClassifier | cv5-1/Validation | 0.9357 | 0.5458 | 0.4730 | 0.3007 | 0.8046 | 0.6050 |
| ExtraTreeClassifier | cv5-2/Validation | 0.9372 | 0.5384 | 0.4512 | 0.2754 | 0.8398 | 0.6677 |
| ExtraTreeClassifier | cv5-3/Validation | 0.9295 | 0.4884 | 0.4121 | 0.2467 | 0.8148 | 0.6070 |
| ExtraTreeClassifier | cv5-4/Validation | 0.9353 | 0.5355 | 0.4473 | 0.2673 | 0.8319 | 0.6478 |
| ExtraTreesClassifier | cv5-0/Validation | 0.9439 | 0.6158 | 0.5443 | 0.3747 | 0.7126 | 0.5680 |
| ExtraTreesClassifier | cv5-1/Validation | 0.9401 | 0.5783 | 0.5023 | 0.3276 | 0.7155 | 0.5715 |
| ExtraTreesClassifier | cv5-2/Validation | 0.9451 | 0.6007 | 0.5211 | 0.3399 | 0.7601 | 0.6178 |
| ExtraTreesClassifier | cv5-3/Validation | 0.9391 | 0.5563 | 0.4836 | 0.3171 | 0.7349 | 0.5559 |
| ExtraTreesClassifier | cv5-4/Validation | 0.9406 | 0.5832 | 0.5050 | 0.3308 | 0.7359 | 0.6089 |
| KNeighborsClassifier | cv5-0/Validation | 0.9082 | 0.2878 | 0.2168 | 0.1066 | 0.8572 | 0.5732 |
| KNeighborsClassifier | cv5-1/Validation | 0.9063 | 0.2882 | 0.2213 | 0.1103 | 0.8497 | 0.5628 |
| KNeighborsClassifier | cv5-2/Validation | 0.9063 | 0.2537 | 0.1892 | 0.0926 | 0.8475 | 0.5559 |
| KNeighborsClassifier | cv5-3/Validation | 0.9031 | 0.2428 | 0.1814 | 0.0872 | 0.8380 | 0.5225 |
| KNeighborsClassifier | cv5-4/Validation | 0.9072 | 0.3084 | 0.2304 | 0.1116 | 0.8654 | 0.6128 |
| MLPClassifier | cv5-0/Validation | 0.9525 | 0.6963 | 0.6411 | 0.4440 | 0.9158 | 0.7779 |
| MLPClassifier | cv5-1/Validation | 0.9509 | 0.6663 | 0.6118 | 0.4176 | 0.9224 | 0.7666 |
| MLPClassifier | cv5-2/Validation | 0.9528 | 0.6522 | 0.5908 | 0.3998 | 0.9186 | 0.7715 |
| MLPClassifier | cv5-3/Validation | 0.9531 | 0.6853 | 0.6233 | 0.4258 | 0.8977 | 0.7522 |
| MLPClassifier | cv5-4/Validation | 0.9546 | 0.7125 | 0.6529 | 0.4545 | 0.9283 | 0.8052 |
| RandomForestClassifier | cv5-0/Validation | 0.9505 | 0.6805 | 0.6212 | 0.4389 | 0.7593 | 0.6220 |
| RandomForestClassifier | cv5-1/Validation | 0.9498 | 0.6729 | 0.6089 | 0.4179 | 0.7336 | 0.6023 |
| RandomForestClassifier | cv5-2/Validation | 0.9521 | 0.6751 | 0.5989 | 0.3984 | 0.7868 | 0.6534 |
| RandomForestClassifier | cv5-3/Validation | 0.9462 | 0.6399 | 0.5758 | 0.3891 | 0.7471 | 0.5959 |
| RandomForestClassifier | cv5-4/Validation | 0.9493 | 0.6839 | 0.6081 | 0.4011 | 0.7720 | 0.6569 |
| RidgeClassifierCV | cv5-0/Validation | 0.9421 | 0.6536 | 0.6127 | 0.4109 | 0.8429 | 0.6484 |
| RidgeClassifierCV | cv5-1/Validation | 0.9440 | 0.6507 | 0.6134 | 0.4061 | 0.8459 | 0.6497 |
| RidgeClassifierCV | cv5-2/Validation | 0.9467 | 0.6541 | 0.6100 | 0.4027 | 0.8390 | 0.6487 |
| RidgeClassifierCV | cv5-3/Validation | 0.9417 | 0.6381 | 0.5975 | 0.3947 | 0.8408 | 0.6402 |
| RidgeClassifierCV | cv5-4/Validation | 0.9433 | 0.6636 | 0.6250 | 0.4155 | 0.8511 | 0.6571 |

#### Statistical Mean of Columns

| Lead Classifier | Accuracy | F_Measure | F_Beta | G_Beta | AUROC | AUPRC |
|-----------------|----------|-----------|--------|--------|-------|-------|
| DecisionTreeClassifier | 0.9452 | 0.6214 | 0.5446 | 0.3546 | 0.8138 | 0.6590 |
| ExtraTreeClassifier | 0.9344 | 0.5264 | 0.4452 | 0.2720 | 0.8208 | 0.6300 |
| ExtraTreesClassifier | 0.9418 | 0.5869 | 0.5113 | 0.3380 | 0.7318 | 0.5844 |
| KNeighborsClassifier | 0.9062 | 0.2762 | 0.2078 | 0.1017 | 0.8516 | 0.5654 |
| MLPClassifier | 0.9528 | 0.6825 | 0.6240 | 0.4283 | 0.9166 | 0.7747 |
| RandomForestClassifier | 0.9496 | 0.6705 | 0.6026 | 0.4091 | 0.7598 | 0.6261 |
| RidgeClassifierCV | 0.9436 | 0.6520 | 0.6117 | 0.4060 | 0.8439 | 0.6488 |


## Multiclass 

TODO: The following algorithms are Scikit learn **Inherently Multiclass** algorithms that do not support multilabel.

```python
sklearn.naive_bayes.BernoulliNB
sklearn.naive_bayes.GaussianNB
sklearn.semi_supervised.LabelPropagation
sklearn.semi_supervised.LabelSpreading
sklearn.discriminant_analysis.LinearDiscriminantAnalysis
sklearn.svm.LinearSVC (setting multi_class=”crammer_singer”)
sklearn.linear_model.LogisticRegression (setting multi_class=”multinomial”)
sklearn.linear_model.LogisticRegressionCV (setting multi_class=”multinomial”)
sklearn.neighbors.NearestCentroid
sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis
sklearn.linear_model.RidgeClassifier
```

## Multiclass as One-Vs-One

```python
sklearn.svm.NuSVC
sklearn.svm.SVC
sklearn.gaussian_process.GaussianProcessClassifier # (setting multi_class = “one_vs_one”)
```

## Multiclass as One-Vs-The-Rest

```python
sklearn.ensemble.GradientBoostingClassifier
sklearn.gaussian_process.GaussianProcessClassifier # (setting multi_class = “one_vs_rest”)
sklearn.svm.LinearSVC # (setting multi_class=”ovr”)
sklearn.linear_model.LogisticRegression # (setting multi_class=”ovr”)
sklearn.linear_model.LogisticRegressionCV #(setting multi_class=”ovr”)
sklearn.linear_model.SGDClassifier
sklearn.linear_model.Perceptron
sklearn.linear_model.PassiveAggressiveClassifier
```

## Errors

TODO: clean up experiment runs code

LinearSVC: very sloooow. Unlikely to be a candidate model.

```text
/home/alex/sandbox/src/git.udia.ca/alex/physionet-challenge-2020/venv/lib/python3.6/site-packages/sklearn/svm/_base.py:947: Converge$
ceWarning: Liblinear failed to converge, increase the number of iterations.
  "the number of iterations.", ConvergenceWarning)
...
/home/alex/sandbox/src/git.udia.ca/alex/physionet-challenge-2020/venv/lib/python3.6/site-packages/sklearn/svm/_base.py:947: Convergen
ceWarning: Liblinear failed to converge, increase the number of iterations.
  "the number of iterations.", ConvergenceWarning)
[INFO]: Took 0:32:36.466003
```

LogisticRegression & LogisticRegressionCV: default settings do not converge per lead
```text
[INFO]: Fitting lead I classifier on training data (5501, 484)...
/home/alex/sandbox/src/git.udia.ca/alex/physionet-challenge-2020/venv/lib/python3.6/site-packages/sklearn/linear_model/_logistic.py:9
40: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
/home/alex/sandbox/src/git.udia.ca/alex/physionet-challenge-2020/venv/lib/python3.6/site-packages/sklearn/linear_model/_logistic.py:9
40: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
```

QuadraticDiscriminantAnalysis
```text
/home/alex/sandbox/src/git.udia.ca/alex/physionet-challenge-2020/venv/lib/python3.6/site-packages/sklearn/discriminant_analysis.py:69
1: UserWarning: Variables are collinear
  warnings.warn("Variables are collinear")
```

RidgeClassifier
```text
/home/alex/sandbox/src/git.udia.ca/alex/physionet-challenge-2020/venv/lib/python3.6/site-packages/sklearn/linear_model/_ridge.py:148:
 LinAlgWarning: Ill-conditioned matrix (rcond=1.6382e-17): result may not be accurate.                                               
  overwrite_a=True).T
```

NuSVC
```text
Traceback (most recent call last):                                                                                                   
  File "main.py", line 27, in main                                                                                                   
    agent.run()                                                                                                                      
  File "/home/alex/sandbox/src/git.udia.ca/alex/physionet-challenge-2020/agents/scikit_learn.py", line 149, in run                   
    self.lead_classifiers[lead].fit(lead_inputs[lead], targets)                                                                      
  File "/home/alex/sandbox/src/git.udia.ca/alex/physionet-challenge-2020/venv/lib/python3.6/site-packages/sklearn/multioutput.py", li
ne 359, in fit                                                                                                                       
    super().fit(X, Y, sample_weight)                                                                                                 
  File "/home/alex/sandbox/src/git.udia.ca/alex/physionet-challenge-2020/venv/lib/python3.6/site-packages/sklearn/multioutput.py", li
ne 170, in fit                                                                                                                       
    for i in range(y.shape[1]))                                                                                                      
  File "/home/alex/sandbox/src/git.udia.ca/alex/physionet-challenge-2020/venv/lib/python3.6/site-packages/joblib/parallel.py", line 9
21, in __call__                                                                                                                      
    if self.dispatch_one_batch(iterator):                                                                                            
  File "/home/alex/sandbox/src/git.udia.ca/alex/physionet-challenge-2020/venv/lib/python3.6/site-packages/joblib/parallel.py", line 7
59, in dispatch_one_batch                                                                                                            
    self._dispatch(tasks)                                                                                                            
  File "/home/alex/sandbox/src/git.udia.ca/alex/physionet-challenge-2020/venv/lib/python3.6/site-packages/joblib/parallel.py", line 7
16, in _dispatch                                                                                                                     
    job = self._backend.apply_async(batch, callback=cb)                                                                              
  File "/home/alex/sandbox/src/git.udia.ca/alex/physionet-challenge-2020/venv/lib/python3.6/site-packages/joblib/_parallel_backends.p
y", line 182, in apply_async                                                                                                         
    result = ImmediateResult(func)                                                                                                   
  File "/home/alex/sandbox/src/git.udia.ca/alex/physionet-challenge-2020/venv/lib/python3.6/site-packages/joblib/_parallel_backends.py", line 549, in __init__
    self.results = batch()
  File "/home/alex/sandbox/src/git.udia.ca/alex/physionet-challenge-2020/venv/lib/python3.6/site-packages/joblib/parallel.py", line $25, in __call__
    for func, args, kwargs in self.items]
  File "/home/alex/sandbox/src/git.udia.ca/alex/physionet-challenge-2020/venv/lib/python3.6/site-packages/joblib/parallel.py", line $25, in <listcomp>
    for func, args, kwargs in self.items]
  File "/home/alex/sandbox/src/git.udia.ca/alex/physionet-challenge-2020/venv/lib/python3.6/site-packages/sklearn/multioutput.py", l$ne 40, in _fit_estimator
    estimator.fit(X, y)
  File "/home/alex/sandbox/src/git.udia.ca/alex/physionet-challenge-2020/venv/lib/python3.6/site-packages/sklearn/svm/_base.py", lin$ 199, in fit
    fit(X, y, sample_weight, solver_type, kernel, random_seed=seed)                                                                 
  File "/home/alex/sandbox/src/git.udia.ca/alex/physionet-challenge-2020/venv/lib/python3.6/site-packages/sklearn/svm/_base.py", lin$ 258, in _dense_fit
    max_iter=self.max_iter, random_seed=random_seed)
  File "sklearn/svm/_libsvm.pyx", line 191, in sklearn.svm._libsvm.fit                                                              
ValueError: specified nu is infeasible
```