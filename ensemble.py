from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv
)
import random

import numpy as np


def sigmoid(x):
    """Apply sigmoid function."""
    return np.exp(x) / (1 + np.exp(x))


def bootstrap(data):
    rand_sample = {'user_id': [], 'question_id': [], 'is_correct': []}
    num_data = len(data['is_correct'])
    for _ in range(num_data):
        i = random.randint(0, num_data - 1)
        rand_sample['user_id'].append(data['user_id'][i])
        rand_sample['question_id'].append(data['question_id'][i])
        rand_sample['is_correct'].append(data['is_correct'][i])
    return rand_sample


def update_theta_beta(data, lr, theta, beta):
    partial_theta = np.zeros(theta.shape[0])
    partial_beta = np.zeros(beta.shape[0])
    for i in range(len(data['is_correct'])):
        user = data['user_id'][i]
        question = data['question_id'][i]
        sig = sigmoid(theta[user] - beta[question])
        partial_theta[user] += data['is_correct'][i] - sig
        partial_beta[question] += -data['is_correct'][i] + sig
    theta += lr * partial_theta
    beta += lr * partial_beta

    return theta, beta


def irt_train(data, lr, iterations):
    theta = np.zeros(len(data['user_id']))
    beta = np.zeros(len(data['question_id']))

    for _ in range(iterations):
        theta, beta = update_theta_beta(data, lr, theta, beta)

    return theta, beta


def irt_predict(data, theta, beta):
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(int(p_a >= 0.5))
    return pred


def ensemble(num_resamples, train_data, val_data, lr, iterations):
    resamples = [bootstrap(train_data) for _ in range(num_resamples)]
    pred_mat = []
    for i in range(num_resamples):
        theta, beta = irt_train(resamples[i], lr, iterations)
        pred_mat.append(irt_predict(val_data, theta, beta))
    pred_mat = np.array(pred_mat)
    avg_pred = (np.sum(pred_mat, axis=0) / num_resamples >= 0.5).astype(int)
    return avg_pred


def ensemble_evaluate():
    train_data = load_train_csv()
    valid_data = load_valid_csv()
    test_data = load_public_test_csv()

    valid_pred = ensemble(3, train_data, valid_data, 0.02, 50)
    test_pred = ensemble(3, train_data, test_data, 0.02, 50)

    valid_acc = np.sum((valid_pred == valid_data['is_correct'])) / len(valid_data['is_correct'])
    test_acc = np.sum((test_pred == test_data['is_correct'])) / len(test_data['is_correct'])
    print('Validation accuracy: {:.6f}\nTest accuracy: {:.6f}'.
          format(valid_acc, test_acc))


if __name__ == '__main__':
    ensemble_evaluate()
