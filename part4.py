import numpy
import pylab as p

from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

def sigmoid(x):
    """Apply sigmoid function."""
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list} -> A sparse matrix {row = question_id, col =
    user_id}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    u = len(theta)  # 542
    q = len(beta)   # 1774
    theta = np.tile(theta, (q, 1))
    beta = np.tile(beta.reshape((-1, 1)), (1, u))
    # print((theta.shape, beta.shape))
    t_minus_b = theta - beta    # 1774 x 542
    sig = sigmoid(t_minus_b)
    # print(data.shape)
    data_minus_ones = data.copy()
    data_minus_ones.data -= 1
    log_lklihood_matrix = data.multiply(np.log(sig + 0.001)) - data_minus_ones.multiply(np.log(1 - sig + 0.001))
    log_lklihood = log_lklihood_matrix.sum()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list} -> A sparse matrix {row = question_id, col =
    user_id}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    u = len(theta)
    q = len(beta)
    theta_copy = np.tile(theta, (q, 1))
    beta_copy = np.tile(beta.reshape((-1, 1)), (1, u))
    sig = sigmoid(theta_copy - beta_copy)
    data_minus_ones = data.copy()
    data_minus_ones.data -= 1
    sig = (data - data_minus_ones).multiply(sig)

    t_dst = np.sum(data - sig, axis=0).reshape((-1, 1))
    b_dst = np.sum(-data + sig, axis=1)
    # print((t_dst.shape, b_dst.shape))
    # print((np.sum(t_dst), np.sum(b_dst)))
    theta = np.diag(theta + lr * t_dst)
    beta = np.diag(beta + lr * b_dst)
    # print((theta.shape, beta.shape))
    # print((np.sum(theta), np.sum(beta)))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, tra_data, lr, iterations):
    """Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list} -> A sparse matrix {row = question_id, col =
    user_id}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    u = data.shape[1]
    q = data.shape[0]
    theta = np.array([1 for i in range(u)])
    beta = np.array([1 for i in range(q)])

    val_acc_lst = []
    tra_acc_lst = []
    tra_nlld_list = []
    val_nlld_list = []

    rows = val_data['question_id']
    cols = val_data['user_id']
    value = val_data['is_correct']
    val_sparse_data = csr_matrix((value, (rows, cols)), shape=(max(rows) + 1, max(cols) + 1))

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        tra_nlld_list.append(neg_lld)
        val_nlld_list.append(neg_log_likelihood(val_sparse_data, theta=theta, beta=beta))
        val_acc_lst.append(score)
        tra_acc_lst.append(evaluate(data=tra_data, theta=theta, beta=beta))
        # print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)
    # TODO: You may change the return values to achieve what you want.
    # return theta, beta, val_acc_lst, tra_acc_lst, val_nlld_list, tra_nlld_list
    return theta, beta, val_acc_lst


def evaluate(data, theta, beta):
    """Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])

def evaluate_test(data, t1, t2 ,t3, b1, b2, b3):
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        pred1 = p_a(t1[u], b1[q])
        pred2 = p_a(t2[u], b2[q])
        pred3 = p_a(t3[u], b3[q])
        p = (pred1 + pred2 + pred3) / 3
        pred.append(p >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])

def p_a(t, b):
    x = (t - b).sum()
    return sigmoid(x)

# def main():
#     train_data = load_train_csv("./data")
#     # You may optionally use the sparse matrix.
#     # sparse_matrix = load_train_sparse("./data")
#     val_data = load_valid_csv("./data")
#     test_data = load_public_test_csv("./data")
#
#     #####################################################################
#     # TODO:                                                             #
#     # Tune learning rate and number of iterations. With the implemented #
#     # code, report the validation and test accuracy.                    #
#     #####################################################################
#     rows = train_data['question_id']
#     cols = train_data['user_id']
#     data = train_data['is_correct']
#     sparse_matrix = csr_matrix((data, (rows, cols)), shape=(max(rows) + 1, max(cols) + 1))
#     itera = 30
#     theta, beta, val_acc_lst, tra_acc_lst, val_nlld_list, tra_nlld_list = irt(sparse_matrix, val_data, train_data, 0.005, itera)
#     # plt.plot([i for i in range(1, itera + 1)], val_nlld_list, label='validation data')
#     # plt.plot([i for i in range(1, itera + 1)], tra_nlld_list, label='training data')
#     # plt.ylabel('Accuracy')
#     # plt.ylabel('Neg log-likelihood')
#     # plt.xlabel('Num of iteration')
#     # plt.legend()
#     # plt.savefig('q2_lld.png')
#
#     #####################################################################
#     #                       END OF YOUR CODE                            #
#     #####################################################################
#
#     #####################################################################
#     # TODO:                                                             #
#     # Implement part (d)                                                #
#     #####################################################################
#     p_1 = []
#     p_2 = []
#     p_3 = []
#     theta = sorted(theta)
#     for i in theta:
#         t_b = i - beta[1100]
#         p_1.append(sigmoid(t_b))
#         t_b = i - beta[565]
#         p_2.append(sigmoid(t_b))
#         t_b = i - beta[48]
#         p_3.append(sigmoid(t_b))
#     plt.plot(theta, p_1, label='j1')
#     plt.plot(theta, p_2, label='j2')
#     plt.plot(theta, p_3, label='j3')
#     plt.xlabel('theta')
#     plt.ylabel('probability of correct response')
#     plt.legend()
#     plt.savefig('q2_d.png')
#     #####################################################################
#     #                       END OF YOUR CODE                            #
#     #####################################################################

def bootstrapping(train_data):
    n = len(train_data['question_id'])
    indices = np.random.choice(n, size=n, replace=True)
    rows = []
    cols = []
    data = []
    for i in indices:
        rows.append(train_data['question_id'][i])
        cols.append(train_data['user_id'][i])
        data.append(train_data['is_correct'][i])
    return csr_matrix((data, (rows, cols)), shape=(max(rows) + 1, max(cols) + 1))


def main():
    train_data = load_train_csv("./data")
    # You may optionally use the sparse matrix.
    # sparse_matrix = load_train_sparse("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    sparse_matrix_1 = bootstrapping(train_data)
    sparse_matrix_2 = bootstrapping(train_data)
    sparse_matrix_3 = bootstrapping(train_data)
    print(sparse_matrix_1.shape)
    print(sparse_matrix_2.shape)
    print(sparse_matrix_3.shape)
    print('boostrapping finished')

    itera = 30
    lr = 0.005
    theta_1, beta_1, v_acc_1 = irt(sparse_matrix_1, val_data, train_data, lr, itera)
    theta_2, beta_2, v_acc_2 = irt(sparse_matrix_2, val_data, train_data, lr, itera)
    theta_3, beta_3, v_acc_3 = irt(sparse_matrix_3, val_data, train_data, lr, itera)

    acc = evaluate_test(test_data, theta_1, theta_2, theta_3, beta_1, beta_2, beta_3)
    print(acc)

    plt.plot(v_acc_1, label='validation 1')
    plt.plot(v_acc_2, label='validation 2')
    plt.plot(v_acc_3, label='validation 3')
    plt.ylabel('Accuracy')
    plt.xlabel('Num of iteration')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

