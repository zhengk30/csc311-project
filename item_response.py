from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)
import random
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """Apply sigmoid function."""
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    log_lklihood = 0.0
    for i in range(len(data['is_correct'])):
        is_correct = data['is_correct'][i]
        sig = sigmoid(theta[data['user_id'][i]] - beta[data['question_id'][i]])
        log_lklihood += is_correct * np.log(sig) + (1 - is_correct) * np.log(1 - sig)
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
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


def irt(data, val_data, lr, iterations):
    """Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, log_likelihoods, accuracies)
    """
    theta = np.zeros(len(data['user_id']))
    beta = np.zeros(len(data['question_id']))

    train_log_likelihoods = []
    valid_log_likelihoods = []

    for _ in range(iterations):
        train_log_likelihoods.append(-neg_log_likelihood(data, theta, beta))
        valid_log_likelihoods.append(-neg_log_likelihood(val_data, theta, beta))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    log_likelihoods = {
        'train_log_likelihoods': train_log_likelihoods,
        'valid_log_likelihoods': valid_log_likelihoods
    }

    return theta, beta, log_likelihoods


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


def main():
    train_data = load_train_csv("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    learning_rate = 0.02
    num_iterations = 50
    theta, beta, log_likes = irt(train_data, val_data, learning_rate, num_iterations)
    iterations = [*range(num_iterations)]
    train_loglikes = log_likes['train_log_likelihoods']
    valid_loglikes = log_likes['valid_log_likelihoods']

    plt.figure(figsize=(10, 6))
    plt.title('Log Likelihood vs. No. Iterations')
    plt.plot(iterations, train_loglikes, label='log likelihood (training set)')
    plt.plot(iterations, valid_loglikes, label='log likelihood (validation set)')
    plt.xlabel('No. Iterations')
    plt.ylabel('Log Likelihood')
    plt.legend()
    plt.savefig('loglikes.png')

    valid_accuracy = evaluate(val_data, theta, beta)
    test_accuracy = evaluate(test_data, theta, beta)
    print('Validation accuracy: {:.6f}\nTest accuracy: {:.6f}'.
          format(valid_accuracy, test_accuracy))

    rand_questions = random.sample(range(beta.shape[0]), 3)
    plt.figure(figsize=(10, 6))
    plt.scatter(theta, [sigmoid(t - beta[rand_questions[0]]) for t in theta], label='q1')
    plt.scatter(theta, [sigmoid(t - beta[rand_questions[1]]) for t in theta], label='q2')
    plt.scatter(theta, [sigmoid(t - beta[rand_questions[2]]) for t in theta], label='q3')
    plt.xlabel('theta')
    plt.ylabel('prob. correct prediction')
    plt.legend()
    plt.savefig('prob_vs_theta.png')


if __name__ == '__main__':
    main()
