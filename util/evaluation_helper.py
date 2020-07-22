import numpy as np

from .evaluate_12ECG_score import (
    compute_accuracy,
    compute_auc,
    compute_beta_measures,
    compute_challenge_metric,
    compute_f_measure,
    is_number,
    load_weights,
    organize_labels_outputs,
)


def train_evaluate_score_batch_helper(data_eval, features, data_cache, loaded_model):
    classes = []
    labels = []
    scores = []

    for k, v in loaded_model.items():
        if not is_number(k):
            continue
        classes.append(str(k))

        if len(v) == 2:
            feat_model, class_model = v
            new_features = feat_model.transform(features)
            labels.append(class_model.predict(new_features).tolist())
            scores.append(class_model.predict_proba(new_features)[:, 1].tolist())
        else:
            labels.append(v.predict(features).tolist())
            scores.append(v.predict_proba(features)[:, 1].tolist())

    labels = np.array(labels).T
    scores = np.array(scores).T
    ground_truth = data_eval["dx"].tolist()
    for dx_i, dx in enumerate(ground_truth):
        ground_truth[dx_i] = [str(dv) for dv in dx]

    return evaluate_score_batch(
        predicted_classes=classes,
        predicted_labels=labels,
        predicted_probabilities=scores,
        raw_ground_truth_labels=ground_truth,
    )


def evaluate_score_batch(
    predicted_classes=[],  # list, len(num_classes), str(code)
    predicted_labels=[],  # shape (num_examples, num_classes), T/F for each code
    predicted_probabilities=[],  # shape (num_examples, num_classes), prob. [0-1] for each code
    raw_ground_truth_labels=[],  # list(('dx1', 'dx2'), ('dx1', 'dx3'), ...)
    weights_file="evaluation-2020/weights.csv",
    normal_class="426783006",
    equivalent_classes=[
        ["713427006", "59118001"],
        ["284470004", "63593006"],
        ["427172004", "17338001"],
    ],
):
    """This is a helper function for getting
    auroc, auprc, accuracy, f_measure, f_beta_measure, g_beta_measure, challenge_metric
    without needing the directories of labels and prediction outputs.
    It is useful for directly calculating the scores given the
    classes, predicted labels, and predicted probabilities.
    """

    label_classes, labels = _load_labels(
        raw_ground_truth_labels,
        normal_class=normal_class,
        equivalent_classes_collection=equivalent_classes,
    )

    output_classes, binary_outputs, scalar_outputs = _load_outputs(
        predicted_classes,
        predicted_labels,
        predicted_probabilities,
        normal_class=normal_class,
        equivalent_classes_collection=equivalent_classes,
    )

    classes, labels, binary_outputs, scalar_outputs = organize_labels_outputs(
        label_classes, output_classes, labels, binary_outputs, scalar_outputs
    )

    weights = load_weights(weights_file, classes)

    # Only consider classes that are scored with the Challenge metric.
    indices = np.any(weights, axis=0)  # Find indices of classes in weight matrix.
    classes = [x for i, x in enumerate(classes) if indices[i]]
    labels = labels[:, indices]
    scalar_outputs = scalar_outputs[:, indices]
    binary_outputs = binary_outputs[:, indices]
    weights = weights[np.ix_(indices, indices)]

    auroc, auprc = compute_auc(labels, scalar_outputs)
    accuracy = compute_accuracy(labels, binary_outputs)
    f_measure = compute_f_measure(labels, binary_outputs)
    f_beta_measure, g_beta_measure = compute_beta_measures(
        labels, binary_outputs, beta=2
    )
    challenge_metric = compute_challenge_metric(
        weights, labels, binary_outputs, classes, normal_class
    )

    return (
        auroc,
        auprc,
        accuracy,
        f_measure,
        f_beta_measure,
        g_beta_measure,
        challenge_metric,
    )


# Load labels from header/label files.
def _load_labels(raw_labels, normal_class, equivalent_classes_collection):
    # raw labels: list of str(dx) tuples
    num_recordings = len(raw_labels)

    # Identify classes.
    classes = set.union(*map(set, raw_labels))
    if normal_class not in classes:
        classes.add(normal_class)
        print(
            "- The normal class {} is not one of the label classes, so it has been automatically added, but please check that you chose the correct normal class.".format(
                normal_class
            )
        )
    classes = sorted([str(c) for c in classes])
    num_classes = len(classes)

    # Use one-hot encoding for labels.
    labels = np.zeros((num_recordings, num_classes), dtype=np.bool)
    for i in range(num_recordings):
        dxs = raw_labels[i]
        for dx in dxs:
            j = classes.index(dx)
            labels[i, j] = 1

    # For each set of equivalent class, use only one class as the representative class for the set and discard the other classes in the set.
    # The label for the representative class is positive if any of the labels in the set is positive.
    remove_classes = list()
    remove_indices = list()
    for equivalent_classes in equivalent_classes_collection:
        equivalent_classes = [x for x in equivalent_classes if x in classes]
        if len(equivalent_classes) > 1:
            # representative_class = equivalent_classes[0]
            other_classes = equivalent_classes[1:]
            equivalent_indices = [classes.index(x) for x in equivalent_classes]
            representative_index = equivalent_indices[0]
            other_indices = equivalent_indices[1:]

            labels[:, representative_index] = np.any(
                labels[:, equivalent_indices], axis=1
            )
            remove_classes += other_classes
            remove_indices += other_indices

    for x in remove_classes:
        classes.remove(x)
    labels = np.delete(labels, remove_indices, axis=1)

    # If the labels are negative for all classes, then change the label for the normal class to positive.
    normal_index = classes.index(normal_class)
    for i in range(num_recordings):
        num_positive_classes = np.sum(labels[i, :])
        if num_positive_classes == 0:
            labels[i, normal_index] = 1

    return classes, labels


def _load_outputs(
    predicted_classes,
    predicted_labels,
    predicted_probabilities,
    normal_class,
    equivalent_classes_collection,
):
    # The outputs should have the following form:
    #
    # diagnosis_1, diagnosis_2, diagnosis_3
    #           0,           1,           1
    #        0.12,        0.34,        0.56
    #
    num_recordings = len(predicted_labels)

    tmp_labels = predicted_classes
    tmp_binary_outputs = predicted_labels.tolist()
    tmp_scalar_outputs = predicted_probabilities.tolist()

    # Identify classes.
    classes = set(tmp_labels)
    if normal_class not in classes:
        classes.add(normal_class)
        print(
            "- The normal class {} is not one of the output classes, so it has been automatically added, but please check that you identified the correct normal class.".format(
                normal_class
            )
        )
    classes = sorted(classes)
    num_classes = len(classes)

    # Use one-hot encoding for binary outputs and the same order for scalar outputs.
    binary_outputs = np.zeros((num_recordings, num_classes), dtype=np.bool)
    scalar_outputs = np.zeros((num_recordings, num_classes), dtype=np.float64)
    for i in range(num_recordings):
        for k, dx in enumerate(tmp_labels):
            j = classes.index(dx)
            binary_outputs[i, j] = tmp_binary_outputs[i][k]
            scalar_outputs[i, j] = tmp_scalar_outputs[i][k]

    # For each set of equivalent class, use only one class as the representative class for the set and discard the other classes in the set.
    # The binary output for the representative class is positive if any of the classes in the set is positive.
    # The scalar output is the mean of the scalar outputs for the classes in the set.
    remove_classes = list()
    remove_indices = list()
    for equivalent_classes in equivalent_classes_collection:
        equivalent_classes = [x for x in equivalent_classes if x in classes]
        if len(equivalent_classes) > 1:
            # representative_class = equivalent_classes[0]
            other_classes = equivalent_classes[1:]
            equivalent_indices = [classes.index(x) for x in equivalent_classes]
            representative_index = equivalent_indices[0]
            other_indices = equivalent_indices[1:]

            binary_outputs[:, representative_index] = np.any(
                binary_outputs[:, equivalent_indices], axis=1
            )
            scalar_outputs[:, representative_index] = np.nanmean(
                scalar_outputs[:, equivalent_indices], axis=1
            )
            remove_classes += other_classes
            remove_indices += other_indices

    for x in remove_classes:
        classes.remove(x)
    binary_outputs = np.delete(binary_outputs, remove_indices, axis=1)
    scalar_outputs = np.delete(scalar_outputs, remove_indices, axis=1)

    # If any of the outputs is a NaN, then replace it with a zero.
    binary_outputs[np.isnan(binary_outputs)] = 0
    scalar_outputs[np.isnan(scalar_outputs)] = 0

    # If the binary outputs are negative for all classes, then change the binary output for the normal class to positive.
    normal_index = classes.index(normal_class)
    for i in range(num_recordings):
        num_positive_classes = np.sum(binary_outputs[i, :])
        if num_positive_classes == 0:
            binary_outputs[i, normal_index] = 1

    return classes, binary_outputs, scalar_outputs
