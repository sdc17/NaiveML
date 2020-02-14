# Naive Bayes
from collections import Counter


def data_loader():
    feature_1 = [1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3]
    feature_2 = ['B', 'A', 'A', 'A', 'B', 'B', 'A', 'A', 'C', 'C', 'B', 'A', 'A', 'C', 'C']
    label = [1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, 1, 1, -1]
    return feature_1, feature_2, label


def train_classifier(feature_1, feature_2, label, alpha):  # alpha for Laplace Smooth
    priori_probability = {}
    statistical_probability = {}
    feature_1_counter = Counter(feature_1)
    feature_2_counter = Counter(feature_2)
    label_counter = Counter(label)
    label_length = len(label)
    feature_1_dimension = len(feature_1_counter)
    feature_2_dimension = len(feature_2_counter)
    label_classes = len(label_counter)

    # priori probability
    for label_item in label_counter.keys():
        priori_probability[str(label_item)] = \
            (label_counter[label_item] + alpha) / (label_length + alpha * label_classes)

    for f1, f2, l in zip(feature_1, feature_2, label):
        statistical_probability[str(f1) + '|' + str(l)] = statistical_probability.get(str(f1) + '|' + str(l), 0) + 1
        statistical_probability[str(f2) + '|' + str(l)] = statistical_probability.get(str(f2) + '|' + str(l), 0) + 1

    for label_item in label_counter.keys():
        for f1_item in feature_1_counter.keys():
            statistical_probability[str(f1_item) + '|' + str(label_item)] = \
                (statistical_probability.get(str(f1_item) + '|' + str(label_item), 0) + alpha) / \
                (label_counter[label_item] + alpha * feature_1_dimension)
        for f2_item in feature_2_counter.keys():
            statistical_probability[str(f2_item) + '|' + str(label_item)] = \
                (statistical_probability.get(str(f2_item) + '|' + str(label_item), 0) + alpha) / \
                (label_counter[label_item] + alpha * feature_2_dimension)

    return priori_probability, statistical_probability


def predict(priori_probability, statistical_probability, tester):
    prob_for_every_class = {}
    for i in priori_probability.keys():
        prob_for_every_class[i] = statistical_probability[str(tester[0]) + '|' + str(i)] * \
            statistical_probability[str(tester[1]) + '|' + str(i)] * priori_probability[i]
    prob_max = 0
    for key, value in prob_for_every_class.items():
        if value > prob_max:
            prob_max = value
            prob_max_class = key

    return prob_max_class, prob_max, prob_for_every_class


if __name__ == '__main__':
    # Example
    feature_1, feature_2, label = data_loader()
    priori_probability, statistical_probability = train_classifier(feature_1, feature_2, label, alpha=1)
    tester = [1, 'C']
    prob_max_class, prob_max, prob_for_every_class = predict(priori_probability, statistical_probability, tester)
    print('feature_1: {}, feature2: {}, prediction class: {}'.format(tester[0], tester[1], prob_max_class))

