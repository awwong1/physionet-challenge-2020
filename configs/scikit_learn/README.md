# Results

Scikit-learn [docs](https://scikit-learn.org/stable/modules/multiclass.html).

- Multiclass Classification: classification task with more than two classes. Each sample can only be labelled as one class.

- Multilabel Classification: classification task labelling each sample with `x` labels from `n_classes` possible classes, where `x` can be 0 to `n_classes` inclusive.
    - All Multiclass Classification algorithms can become multilabel classification algorithms using the [MultiOutputClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html#sklearn.multioutput.MultiOutputClassifier).

## Multilabel Algorithms

The following algorithms are Scikit learn Multilabel supporting algorithms.
Each lead is evaluates separately, then the results are stacked together and passed to a `RandomForestClassifier` ensemble.

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

## Multiclass 

TODO: The following algorithms are Scikit learn **Inherently Multiclass** algorithms that do not support multilabel.

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