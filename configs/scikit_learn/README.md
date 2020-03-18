# Results

Scikit-learn [docs](https://scikit-learn.org/stable/modules/multiclass.html).

- Multiclass Classification: classification task with more than two classes. Each sample can only be labelled as one class.

- Multilabel Classification: classification task labelling each sample with `x` labels from `n_classes` possible classes, where `x` can be 0 to `n_classes` inclusive.
    - All Multiclass Classification algorithms can become multilabel classification algorithms using the [MultiOutputClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html#sklearn.multioutput.MultiOutputClassifier).

## Multilabel Algorithms

The following algorithms are Scikit learn Multilabel supporting algorithms.

```
    sklearn.tree.DecisionTreeClassifier
    sklearn.tree.ExtraTreeClassifier
    sklearn.ensemble.ExtraTreesClassifier
    sklearn.neighbors.KNeighborsClassifier
    sklearn.neural_network.MLPClassifier
    sklearn.neighbors.RadiusNeighborsClassifier
    sklearn.ensemble.RandomForestClassifier
    sklearn.linear_model.RidgeClassifierCV
```

### DecisionTree
| Dataset | Accuracy | F_Measure | F_Beta | G_Beta | AUROC | AUPRC |
|-------------------|-------|-------|-------|-------|-------|-------|
| cv-0/Training Set | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| cv-1/Training Set | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| cv-2/Training Set | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| cv-3/Training Set | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| cv-4/Training Set | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| cv-0/Validate Set | 0.922 | 0.608 | 0.618 | 0.379 | 0.000 | 0.000 |
| cv-1/Validate Set | 0.915 | 0.581 | 0.591 | 0.357 | 0.000 | 0.000 |
| cv-2/Validate Set | 0.918 | 0.569 | 0.571 | 0.343 | 0.000 | 0.000 |
| cv-3/Validate Set | 0.917 | 0.575 | 0.577 | 0.354 | 0.000 | 0.000 |
| cv-4/Validate Set | 0.923 | 0.617 | 0.624 | 0.384 | 0.000 | 0.000 |

### ExtraTreeClassifier
| Dataset | Accuracy | F_Measure | F_Beta | G_Beta | AUROC | AUPRC |
|-------------------|-------|-------|-------|-------|-------|-------|
| cv-0/Training Set | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| cv-1/Training Set | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| cv-2/Training Set | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| cv-3/Training Set | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| cv-4/Training Set | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| cv-0/Validate Set | 0.893 | 0.477 | 0.482 | 0.257 | 0.000 | 0.000 |
| cv-1/Validate Set | 0.892 | 0.466 | 0.471 | 0.249 | 0.001 | 0.000 |
| cv-2/Validate Set | 0.896 | 0.498 | 0.501 | 0.273 | 0.000 | 0.000 |
| cv-3/Validate Set | 0.905 | 0.511 | 0.516 | 0.287 | 0.000 | 0.000 |
| cv-4/Validate Set | 0.893 | 0.495 | 0.498 | 0.280 | 0.000 | 0.000 |

### ExtraTreesClassifier
| Dataset | Accuracy | F_Measure | F_Beta | G_Beta | AUROC | AUPRC |
|-------------------|-------|-------|-------|-------|-------|-------|
| cv-0/Training Set | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| cv-1/Training Set | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| cv-2/Training Set | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| cv-3/Training Set | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| cv-4/Training Set | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| cv-0/Validate Set | 0.945 | 0.586 | 0.520 | 0.358 | 0.846 | 0.700 |
| cv-1/Validate Set | 0.943 | 0.570 | 0.505 | 0.347 | 0.838 | 0.702 |
| cv-2/Validate Set | 0.946 | 0.576 | 0.506 | 0.342 | 0.873 | 0.719 |
| cv-3/Validate Set | 0.942 | 0.543 | 0.473 | 0.313 | 0.837 | 0.672 |
| cv-4/Validate Set | 0.943 | 0.575 | 0.502 | 0.340 | 0.842 | 0.715 |

### KNeighborsClassifier
| Dataset | Accuracy | F_Measure | F_Beta | G_Beta | AUROC | AUPRC |
|-------------------|-------|-------|-------|-------|-------|-------|
| cv-0/Training Set | 0.922 | 0.467 | 0.410 | 0.234 | 0.847 | 0.511 |
| cv-1/Training Set | 0.921 | 0.471 | 0.412 | 0.234 | 0.842 | 0.503 |
| cv-2/Training Set | 0.921 | 0.467 | 0.412 | 0.236 | 0.844 | 0.508 |
| cv-3/Training Set | 0.923 | 0.485 | 0.429 | 0.248 | 0.842 | 0.508 |
| cv-4/Training Set | 0.922 | 0.468 | 0.414 | 0.237 | 0.848 | 0.510 |
| cv-0/Validate Set | 0.897 | 0.358 | 0.320 | 0.171 | 0.575 | 0.288 |
| cv-1/Validate Set | 0.894 | 0.327 | 0.297 | 0.154 | 0.569 | 0.267 |
| cv-2/Validate Set | 0.899 | 0.317 | 0.283 | 0.150 | 0.563 | 0.267 |
| cv-3/Validate Set | 0.893 | 0.315 | 0.280 | 0.142 | 0.521 | 0.234 |
| cv-4/Validate Set | 0.897 | 0.324 | 0.281 | 0.143 | 0.601 | 0.279 |

### MLPClassifier
| Dataset | Accuracy | F_Measure | F_Beta | G_Beta | AUROC | AUPRC |
|-------------------|-------|-------|-------|-------|-------|-------|
| cv-0/Training Set | 0.889 | 0.083 | 0.055 | 0.023 | 0.500 | 0.123 |
| cv-1/Training Set | 0.895 | 0.137 | 0.096 | 0.044 | 0.513 | 0.127 |
| cv-2/Training Set | 0.891 | 0.144 | 0.108 | 0.056 | 0.486 | 0.121 |
| cv-3/Training Set | 0.894 | 0.144 | 0.100 | 0.046 | 0.497 | 0.120 |
| cv-4/Training Set | 0.903 | 0.206 | 0.153 | 0.076 | 0.512 | 0.121 |
| cv-0/Validate Set | 0.882 | 0.049 | 0.036 | 0.015 | 0.494 | 0.118 |
| cv-1/Validate Set | 0.883 | 0.079 | 0.062 | 0.027 | 0.495 | 0.122 |
| cv-2/Validate Set | 0.885 | 0.095 | 0.074 | 0.035 | 0.486 | 0.116 |
| cv-3/Validate Set | 0.879 | 0.071 | 0.052 | 0.022 | 0.482 | 0.115 |
| cv-4/Validate Set | 0.889 | 0.151 | 0.121 | 0.058 | 0.500 | 0.118 |

### RadiusNeighborsClassifier
| Dataset | Accuracy | F_Measure | F_Beta | G_Beta | AUROC | AUPRC |
|-------------------|-------|-------|-------|-------|-------|-------|
| cv-0/Training Set | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| cv-1/Training Set | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| cv-2/Training Set | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| cv-3/Training Set | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| cv-4/Training Set | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |

```
ValueError: No neighbors found for test samples array([   0,    1,    2, ..., 1370, 1371, 1372]), you can try using larger radius, giving a label for outliers, or considering removing them from your dataset.
```

### RandomForestClassifier
| Dataset | Accuracy | F_Measure | F_Beta | G_Beta | AUROC | AUPRC |
|-------------------|-------|-------|-------|-------|-------|-------|
| cv-0/Training Set | 1.000 | 1.000 | 0.999 | 0.999 | 0.973 | 0.973 |
| cv-1/Training Set | 1.000 | 1.000 | 1.000 | 0.999 | 0.971 | 0.971 |
| cv-2/Training Set | 1.000 | 1.000 | 1.000 | 1.000 | 0.971 | 0.971 |
| cv-3/Training Set | 1.000 | 1.000 | 1.000 | 1.000 | 0.966 | 0.966 |
| cv-4/Training Set | 1.000 | 1.000 | 1.000 | 1.000 | 0.972 | 0.972 |
| cv-0/Validate Set | 0.951 | 0.647 | 0.581 | 0.403 | 0.956 | 0.819 |
| cv-1/Validate Set | 0.949 | 0.625 | 0.561 | 0.391 | 0.955 | 0.815 |
| cv-2/Validate Set | 0.952 | 0.630 | 0.561 | 0.385 | 0.945 | 0.795 |
| cv-3/Validate Set | 0.950 | 0.618 | 0.550 | 0.379 | 0.941 | 0.785 |
| cv-4/Validate Set | 0.950 | 0.644 | 0.570 | 0.390 | 0.953 | 0.826 |

### RidgeClassifierCV
| Dataset | Accuracy | F_Measure | F_Beta | G_Beta | AUROC | AUPRC |
|-------------------|-------|-------|-------|-------|-------|-------|
| cv-0/Training Set | 0.994 | 0.968 | 0.955 | 0.893 | 0.000 | 0.000 |
| cv-1/Training Set | 0.994 | 0.970 | 0.956 | 0.896 | 0.000 | 0.000 |
| cv-2/Training Set | 0.992 | 0.960 | 0.947 | 0.872 | 0.000 | 0.000 |
| cv-3/Training Set | 0.993 | 0.968 | 0.956 | 0.894 | 0.000 | 0.000 |
| cv-4/Training Set | 0.994 | 0.968 | 0.956 | 0.893 | 0.000 | 0.000 |
| cv-0/Validate Set | 0.895 | 0.480 | 0.464 | 0.242 | 0.000 | 0.000 |
| cv-1/Validate Set | 0.894 | 0.454 | 0.441 | 0.226 | 0.000 | 0.000 |
| cv-2/Validate Set | 0.898 | 0.461 | 0.447 | 0.235 | 0.000 | 0.000 |
| cv-3/Validate Set | 0.895 | 0.474 | 0.458 | 0.237 | 0.000 | 0.000 |
| cv-4/Validate Set | 0.893 | 0.465 | 0.451 | 0.235 | 0.000 | 0.000 |


## Multiclass 

The following algorithms are Scikit learn **Inherently Multiclass** algorithms that do not support multilabel.

```
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