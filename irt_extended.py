from utils import *
from datetime import datetime as dt

import ast
import csv
import os
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta, qinfo, fsmat, c):
    log_like = 0.0
    fmat = np.dot(fsmat, qinfo.transpose()) / np.sum(qinfo, axis=1)
    for i in range(len(data['is_correct'])):
        user = data['user_id'][i]
        question = data['question_id'][i]
        is_correct = data['is_correct'][i]
        sig = c + (1 - c) * sigmoid(theta[user] - beta[question] + fmat[user][question])
        log_like += is_correct * np.log(sig) + (1 - is_correct) * np.log(1 - sig)

    return -log_like


def load_dimensions(base_path='./data'):
    element_poll = set()
    path = os.path.join(base_path, 'question_meta.csv')
    if not os.path.exists(path):
        raise Exception('ERROR: path {} does not exist.'.format(path))
    with open(path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            element_poll.add(row[0])

    num_questions = len(element_poll)
    element_poll.clear()

    path = os.path.join(base_path, 'subject_meta.csv')
    if not os.path.exists(path):
        raise Exception('ERROR: path {} does not exist.'.format(path))
    with open(path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            element_poll.add(row[0])

    num_subjects = len(element_poll)
    element_poll.clear()

    path = os.path.join(base_path, 'student_meta.csv')
    if not os.path.exists(path):
        raise Exception('ERROR: path {} does not exist.'.format(path))
    with open(path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            element_poll.add(row[0])

    num_students = len(element_poll)

    return num_students, num_questions, num_subjects


num_students, num_questions, num_subjects = load_dimensions()


def load_question_info(base_path='./data'):
    """Return a num_questions-by-num_subjects matrix where the (j, s) entry
    represents whether question j belongs to subject s.

    :param base_path: String
    :return: 2D np.array
    """
    path = os.path.join(base_path, 'question_meta.csv')
    if not os.path.exists(path):
        raise Exception('ERROR: path {} does not exist.'.format(path))

    question_info = np.zeros((num_questions, num_subjects))
    with open(path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            question_id = int(row[0])
            subject = ast.literal_eval(row[1])
            question_info[question_id][subject] = 1

    return question_info


def load_fsmat(question_info, train_matrix, zero_train_matrix):
    """Load the matrix of num_students rows and num_subjects columns where
    the (i, s) element represents the familiarity of student i toward subject
    s.

    :param question_info:
    :param train_matrix:
    :param zero_train_matrix:
    :return:
    """
    correct_subjects = np.dot(zero_train_matrix, question_info)
    train_copy = train_matrix.copy()
    train_copy[train_copy == 0] = 1
    train_copy[np.isnan(train_copy)] = 0
    num_subjects = np.dot(train_copy, question_info)
    num_subjects[num_subjects == 0] = 10e-6
    return correct_subjects / num_subjects


def load_student_age(base_path='./data'):
    path = os.path.join(base_path, 'student_meta.csv')
    if not os.path.exists(path):
        raise Exception('ERROR: path {} does not exist.'.format(path))

    student_info = {'user_id': [], 'age': []}
    with open(path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            user = int(row[0])
            dob = row[2].split()
            age = 1 if len(dob) == 0 else dt.now().year - int(dob[0].split('-')[0])
            student_info['user_id'].append(user)
            student_info['age'].append(age)

    return student_info


def update_theta_beta_fsmat(data, data_matrix, data_zero_out, lr, theta, beta, qinfo, fsmat, c):
    num_subjects_per_question = np.count_nonzero(qinfo, axis=1)
    fmat = np.dot(fsmat, qinfo.transpose()) / num_subjects_per_question

    partial_theta = np.zeros(theta.shape[0])
    partial_beta = np.zeros(beta.shape[0])
    for i in range(len(data['is_correct'])):
        user = data['user_id'][i]
        question = data['question_id'][i]
        is_correct = data['is_correct'][i]
        sig = sigmoid(theta[user] - beta[question] + fmat[user][question])
        numerator = (1 - c) * sig * (1 - sig)
        pos_denominator = c + (1 - c) * sig
        neg_denominator = (1 - c) - (1 - c) * sig

        partial_theta[user] += is_correct * numerator / pos_denominator - \
                                (1 - is_correct) * numerator / neg_denominator
        partial_beta[question] += -is_correct * numerator / pos_denominator + \
                                (1 - is_correct) * numerator / neg_denominator

    theta_mat = np.dot(np.expand_dims(theta, axis=1), np.ones((1, num_questions)))
    beta_mat = np.dot(np.ones((num_students, 1)), np.expand_dims(beta, axis=0))
    probs = c + (1 - c) * sigmoid((theta_mat - beta_mat + fmat))
    probs[np.isnan(data_matrix)] = 0
    partial_fsmat = np.dot(
        (1 - c) * (probs - data_zero_out),
        1 / np.dot(np.expand_dims(num_subjects_per_question, axis=1), np.ones((1, num_subjects)))
    )  # Use chain rule to compute the partial of loss with respect to fsmat.

    #     # update alpha
    #     x = (theta_mat - beta_mat + alpha_mat) * k_mat
    #     prob = sigmoid(x) * (1 - c) + c
    #     prob[nan_mask] = 0
    #     # d (alpha_mat) / d (alpha) = 1 / (# subjects in each question)
    #     alpha_mat_de_alpha = 1 / np.expand_dims(np.sum(q_meta, axis=1), axis=1) @ np.ones((1, S))
    #     # d (loss) / d (alpha_mat) = ((prob - correct) * k) * (1-c)
    #     l_de_alpha_mat = ((prob - zero_train_matrix) * k_mat) * (1 - c)
    #
    #     alpha -= lr * (l_de_alpha_mat @ alpha_mat_de_alpha + lambd * alpha)
    theta += lr * partial_theta
    beta += lr * partial_beta
    fsmat += lr * partial_fsmat

    return theta, beta, fsmat


def irt(train_data, train_data_matrix, zero_data_matrix, val_data, qinfo, fsmat, lr, iterations, c):
    theta = np.zeros(num_students)
    beta = np.zeros(num_questions)

    train_log_likelihoods = []
    valid_log_likelihoods = []

    for _ in range(iterations):
        train_log_likelihoods.append(-neg_log_likelihood(train_data, theta, beta, qinfo, fsmat, c))
        valid_log_likelihoods.append(-neg_log_likelihood(val_data, theta, beta, qinfo, fsmat, c))
        theta, beta, fsmat = update_theta_beta_fsmat(
            train_data, train_data_matrix, zero_data_matrix, lr, theta, beta, qinfo, fsmat, c
        )

    log_likelihoods = {
        'train_log_likelihoods': train_log_likelihoods,
        'valid_log_likelihoods': valid_log_likelihoods
    }

    return theta, beta, fsmat, log_likelihoods


def evaluate(data, theta, beta, qinfo, fsmat, c):
    fmat = np.dot(fsmat, qinfo.transpose()) / np.count_nonzero(qinfo, axis=1)
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q] + fmat[u][q]).sum()
        p_a = c + (1 - c) * sigmoid(x)
        pred.append(int(p_a >= 0.5))
    return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])


def main():
    train_data = load_train_csv()
    val_data = load_valid_csv()
    test_data = load_public_test_csv()
    train_matrix = load_train_sparse().toarray()
    zero_train_matrix = train_matrix.copy()
    zero_train_matrix[np.isnan(train_matrix)] = 0

    qinfo = load_question_info()
    fsmat = load_fsmat(qinfo, train_matrix, zero_train_matrix)

    lr = 0.005
    num_iterations = 50
    # guessing_param = 0.12
    # theta, beta, fsmat, log_likes = irt(
    #     train_data, train_matrix, zero_train_matrix, val_data,
    #     qinfo, fsmat, lr, num_iterations, guessing_param
    # )
    # val_acc = evaluate(val_data, theta, beta, qinfo, fsmat, guessing_param)
    # print(val_acc)
    # test_acc = evaluate(test_data, theta, beta, qinfo, fsmat, guessing_param)
    # print(test_acc)
#     guessing_params = [0.1, 0.12, 0.14, 0.16, 0.18, 0.20, 0.25, 0.33]
#     iterations = [*range(num_iterations)]
#     train_accs, valid_accs = [], []
#     for gp in guessing_params:
#         theta, beta, fsmat, log_likes = irt(
#             train_data, train_matrix, zero_train_matrix, val_data,
#             qinfo, fsmat, lr, num_iterations, gp
#         )
#
#         # train_loglikes = log_likes['train_log_likelihoods']
#         # valid_loglikes = log_likes['valid_log_likelihoods']
#
#         # plt.plot(iterations, train_loglikes, label='log likelihood (training set):' + 'gp={}'.format(str(gp)))
#         # plt.plot(iterations, valid_loglikes, label='log likelihood (validation set):' + 'gp={}'.format(str(gp)))
#
# ##        train_accuracy = evaluate(train_data, theta, beta, qinfo, fsmat)
#         valid_accuracy = evaluate(val_data, theta, beta, qinfo, fsmat, gp)
# ##        print('====================Guessing Parameter={}===================='.format(gp))
#         print('Guessing parameter: {:.6f}\tValidation accuracy: {:.6f}'.
#               format(gp, valid_accuracy))
    optimal_gp = 0.12
    theta, beta, fsmat, log_likes = irt(
        train_data, train_matrix, zero_train_matrix, val_data,
        qinfo, fsmat, lr, num_iterations, optimal_gp
    )
    print(evaluate(val_data, theta, beta, qinfo, fsmat, optimal_gp))
    print(evaluate(test_data, theta, beta, qinfo, fsmat, optimal_gp))

    plt.figure(figsize=(10, 6))
    plt.title('Log Likelihood vs. No. Iterations')
    plt.plot([*range(num_iterations)], log_likes['train_log_likelihoods'], label='training ll.')
    plt.plot([*range(num_iterations)], log_likes['valid_log_likelihoods'], label='validation ll.')
    plt.xlabel('iteration')
    plt.ylabel('log likelihood')
    plt.legend()
    plt.savefig('extended_irt.png')


if __name__ == '__main__':
    main()
