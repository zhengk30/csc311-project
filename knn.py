import numpy as np
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
)


def knn_impute_by_user(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    mat = nbrs.fit_transform(matrix.transpose())
    acc = sparse_matrix_evaluate(valid_data, mat.transpose())
    print('Validation Accuracy: {}'.format(acc))
    return acc


def _knn_evaluate(k_vals, sparse_matrix, val_data, test_data, user_based):
    """A helper function used to evaluate the performance of kNN filtering
    for different k's in k_vals. The evaluation differs based on whether
    the filtering is user-based or question-based. The final test accuracy
    will also be computed.
    """
    valid_accuracies = []
    cfilter = knn_impute_by_user if user_based else knn_impute_by_item
    figsave_path = 'knn_user.png' if user_based else 'knn_item.png'
    for k in k_vals:
        acc = cfilter(sparse_matrix, val_data, k)
        valid_accuracies.append(acc)

    plt.figure(figsize=(10, 6))
    plt.plot(k_vals, valid_accuracies)
    plt.xlabel('k_val')
    plt.ylabel('validation_accuracy')
    plt.title('Validation Accuracy vs. k')
    plt.savefig(figsave_path)

    k_star = k_vals[valid_accuracies.index(max(valid_accuracies))]
    nbrs = KNNImputer(n_neighbors=k_star)
    which_sparse = sparse_matrix if user_based else sparse_matrix.transpose()
    which_test = nbrs.fit_transform(which_sparse)
    which_result = which_test if user_based else which_test.transpose()
    test_acc = sparse_matrix_evaluate(test_data, which_result)

    print('Optimal k: {}'.format(k_star))
    print('Test accuracy: {}'.format(test_acc))


def main():
    sparse_matrix = load_train_sparse("./data").toarray()
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    k_vals = [*range(1, 27, 5)]
    print('----------User-Based Filtering----------')
    _knn_evaluate(k_vals, sparse_matrix, val_data, test_data, True)
    print('\n----------Item-Based Filtering----------')
    _knn_evaluate(k_vals, sparse_matrix, val_data, test_data, False)


if __name__ == "__main__":
    main()
