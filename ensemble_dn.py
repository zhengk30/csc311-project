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
    return np.exp(x) / (1 + np.exp(x))


def update_theta_beta(data, lr, theta, beta):
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
    theta = np.diag(theta + lr * t_dst)
    beta = np.diag(beta + lr * b_dst)
    return theta, beta


def irt(data, lr, iterations):
    u = data.shape[1]
    q = data.shape[0]
    theta = np.array([1 for i in range(u)])
    beta = np.array([1 for i in range(q)])
    for i in range(iterations):
        theta, beta = update_theta_beta(data, lr, theta, beta)
    return theta, beta


def evaluate_single(data, t, b):
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        pred.append(p_a(t[u], b[q]) >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])


def evaluate_test(data, t1, t2 ,t3, b1, b2, b3):
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        pred1 = int(p_a(t1[u], b1[q]) >= 0.5)
        pred2 = int(p_a(t2[u], b2[q]) >= 0.5)
        pred3 = int(p_a(t3[u], b3[q]) >= 0.5)
        p = (pred1 + pred2 + pred3) / 3
        pred.append(int(p >= 0.5))
    return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])

def p_a(t, b):
    x = (t - b).sum()
    return sigmoid(x)

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
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    sparse_matrix_1 = bootstrapping(train_data)
    sparse_matrix_2 = bootstrapping(train_data)
    sparse_matrix_3 = bootstrapping(train_data)

    itera = 50
    lr = 0.02
    theta_1, beta_1 = irt(sparse_matrix_1, lr, itera)
    theta_2, beta_2 = irt(sparse_matrix_2, lr, itera)
    theta_3, beta_3 = irt(sparse_matrix_3, lr, itera)

    valid_acc = evaluate_single(val_data, theta_1, beta_1)
    print(f'validation accuracy 1: {valid_acc}')
    valid_acc = evaluate_single(val_data, theta_2, beta_2)
    print(f'validation accuracy 2: {valid_acc}')
    valid_acc = evaluate_single(val_data, theta_3, beta_3)
    print(f'validation accuracy 3: {valid_acc}')

    print()

    valid_acc = evaluate_test(val_data, theta_1, theta_2, theta_3, beta_1, beta_2, beta_3)
    print(f'validation accuracy: {valid_acc}')

    test_acc = evaluate_test(test_data, theta_1, theta_2, theta_3, beta_1, beta_2, beta_3)
    print(f'test accuracy: {test_acc}')


if __name__ == "__main__":
    main()
