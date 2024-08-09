import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch
from scipy.sparse import csr_matrix

from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    load_train_csv
)


def load_data(base_path="./data"):
    """Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    train_dict = load_train_csv()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, train_dict, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for h and g.                              #
        #####################################################################
        encode = torch.sigmoid(self.g(inputs))
        out = torch.sigmoid(self.h(encode))
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model, lr, lamb, train_data, zero_train_data, train_dict, valid_data, num_epoch):
    """Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param train_dict: Dict
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function.

    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]
    # max_val_acc = -1
    valid_accs = []
    train_accs = []

    rows = max(valid_data["user_id"]) + 1
    cols = max(valid_data["question_id"]) + 1
    valid_sparse = torch.zeros((rows, cols), dtype=torch.float32)
    for r, c, v in zip(valid_data["user_id"], valid_data["question_id"], valid_data["is_correct"]):
        valid_sparse[r, c] = v

    zero_valid_matrix = valid_sparse.clone()
    zero_valid_matrix = torch.nan_to_num(valid_sparse, nan=0.0)

    train_loss_lst = []
    valid_loss_lst = []

    for epoch in range(0, num_epoch):
        train_loss = 0.0
        valid_loss = 0.0

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[nan_mask] = output[nan_mask]

            tra_loss = torch.sum((output - target) ** 2.0)
            tra_loss.backward()

            train_loss += tra_loss.item()

            # For validation
            inputs = Variable(zero_valid_matrix[user_id]).unsqueeze(0)
            target = inputs.clone()

            output = model(inputs)
            nan_mask = np.isnan(valid_sparse[user_id].unsqueeze(0).numpy())
            target[nan_mask] = output[nan_mask]

            val_loss = torch.sum((output - target) ** 2.0)

            valid_loss += val_loss.item()
            optimizer.step()

        train_loss_lst.append(train_loss)
        valid_loss_lst.append(valid_loss)

        # train_loss += (lamb / 2) * (model.get_weight_norm())

        valid_acc = evaluate(model, zero_train_data, valid_data)
        valid_accs.append(valid_acc)
        train_acc = evaluate(model, zero_train_data, train_dict)
        train_accs.append(train_acc)
        print(
            "Epoch: {} \tTraining Cost: {:.6f} \tValidation Cost: {:.6f} \t  Train Acc: {} \t  Valid Acc: {}".format(
                epoch, train_loss, valid_loss, train_acc, valid_acc
            )
        )
        # max_val_acc = max(max_val_acc, valid_acc)
    return valid_accs, train_accs, valid_loss_lst, train_loss_lst
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    zero_train_matrix, train_matrix, train_dict, valid_data, test_data = load_data()

    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    # Set model hyperparameters.
    # k_lst = [10, 50, 100, 200, 500]
    k = 50
    # lamb_lst = [0.001, 0.01, 0.1, 1]
    lamb_lst = [1]
    # k_star = -1
    # max_val_acc = -1
    val_acc = []
    tra_acc = []
    val_loss = []
    tra_loss = []
    num_q = len(zero_train_matrix[0])

    model = AutoEncoder(num_q, k)
    for lamb in lamb_lst:
        # Set optimization hyperparameters.
        lr = 0.01
        num_epoch = 41

        val_acc, tra_acc, val_loss, tra_loss = train(model, lr, lamb, train_matrix, zero_train_matrix, train_dict, valid_data, num_epoch)
        # Next, evaluate your network on validation/test data
        # if val_acc > max_val_acc:
        #     k_star = k
        #     max_val_acc = val_acc
    #     print((lamb, evaluate(model, zero_train_matrix, test_data), evaluate(model, zero_train_matrix, valid_data)))
    # print("k*: {} \tHighest Valid Acc: {}".format(k_star, max_val_acc))

    plt.figure()
    plt.plot(val_acc, label='Validation Accuracy')
    plt.plot(tra_acc, label='Training Accuracy')
    plt.xlabel('Num of Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('q3_d_acc.png')

    plt.figure()
    plt.plot(val_loss, label='Validation loss')
    plt.plot(tra_loss, label='Training loss')
    plt.xlabel('Num of Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('q3_d_loss.png')

    print(evaluate(model, zero_train_matrix, test_data))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
