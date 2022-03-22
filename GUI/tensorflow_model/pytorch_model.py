import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import pandas as pd
# from IPython.display import display


class DQN(nn.Module):

    def __init__(self, outputs):
        super(DQN, self).__init__()
        # 6 by 7, 10 by 11
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.conv6 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(32, 32, kernel_size=5, padding=2)

        linear_input_size = 6 * 7 * 32
        self.MLP1 = nn.Linear(linear_input_size, 50)
        self.MLP2 = nn.Linear(50, 50)
        self.MLP3 = nn.Linear(50, 50)
        self.MLP4 = nn.Linear(50, outputs)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.conv5(x))
        x = F.leaky_relu(self.conv6(x))
        x = F.leaky_relu(self.conv7(x))
        # flatten the feature vector except batch dimension
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.MLP1(x))
        x = F.leaky_relu(self.MLP2(x))
        x = F.leaky_relu(self.MLP3(x))
        return self.MLP4(x)


def load_model(path = 'DQN_plainCNN.pth', device="cpu"):
    model = DQN(7)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


class connect_x:

    def __init__(self):
        self.board_height = 6
        self.board_width = 7
        self.board_state = np.zeros([self.board_height, self.board_width], dtype=np.int8)
        self.players = {'p1': 1, 'p2': 2}
        self.isDone = False
        self.reward = {'win': 1, 'draw': 0.5, 'lose': -1}

    # def render(self):
    #     rendered_board_state = self.board_state.copy().astype(str)
    #     rendered_board_state[self.board_state == 0] = ' '
    #     rendered_board_state[self.board_state == 1] = 'O'
    #     rendered_board_state[self.board_state == 2] = 'X'
    #     display(pd.DataFrame(rendered_board_state))

    def reset(self):
        self.__init__()

    def get_available_actions(self):
        available_cols = []
        for j in range(self.board_width):
            if np.sum([self.board_state[:, j] == 0]) != 0:
                available_cols.append(j)
        return available_cols

    def check_game_done(self, player):
        if player == 'p1':
            check = '1 1 1 1'
        else:
            check = '2 2 2 2'

        # check vertically then horizontally
        for j in range(self.board_width):
            if check in np.array_str(self.board_state[:, j]):
                self.isDone = True
        for i in range(self.board_height):
            if check in np.array_str(self.board_state[i, :]):
                self.isDone = True

        # check left diagonal and right diagonal
        for k in range(0, self.board_height - 4 + 1):
            left_diagonal = np.array([self.board_state[k + d, d] for d in \
                                      range(min(self.board_height - k, min(self.board_height, self.board_width)))])
            right_diagonal = np.array([self.board_state[d + k, self.board_width - d - 1] for d in \
                                       range(min(self.board_height - k, min(self.board_height, self.board_width)))])
            if check in np.array_str(left_diagonal) or check in np.array_str(right_diagonal):
                self.isDone = True
        for k in range(1, self.board_width - 4 + 1):
            left_diagonal = np.array([self.board_state[d, d + k] for d in \
                                      range(min(self.board_width - k, min(self.board_height, self.board_width)))])
            right_diagonal = np.array([self.board_state[d, self.board_width - 1 - k - d] for d in \
                                       range(min(self.board_width - k, min(self.board_height, self.board_width)))])
            if check in np.array_str(left_diagonal) or check in np.array_str(right_diagonal):
                self.isDone = True

        if self.isDone:
            return self.reward['win']
        # check for draw
        elif np.sum([self.board_state == 0]) == 0:
            self.isDone = True
            return self.reward['draw']
        else:
            return 0.

    def make_move(self, a, player):
        # check if move is valid
        if a in self.get_available_actions():
            i = np.sum([self.board_state[:, a] == 0]) - 1
            self.board_state[i, a] = self.players[player]
        else:
            print('Move is invalid')
            self.render()

        reward = self.check_game_done(player)

        # give feedback as new state and reward
        return self.board_state.copy(), reward


def select_action(model, state, available_actions, steps_done=None, training=True):
    # batch and color channel
    state = torch.tensor(state, dtype=torch.float, device="cpu").unsqueeze(dim=0).unsqueeze(dim=0)

    with torch.no_grad():
        # action recommendations from policy net
        r_actions = model(state)[0, :]
        state_action_values = [r_actions[action] for action in available_actions]
        argmax_action = np.argmax(state_action_values)
        greedy_action = available_actions[argmax_action]
        return greedy_action


def demo(model):
    env.reset()
    env.render()

    while not env.isDone:
        state = env.board_state.copy()
        available_actions = env.get_available_actions()
        action = select_action(model, state, available_actions, training=False)
        print(f"P1 chooses action {action}")
        # trained agent's move is denoted by O
        state, reward = env.make_move(action, 'p1')
        env.render()

        if reward == 1:
            break

        available_actions = env.get_available_actions()
        action = select_action(model, state, available_actions, training=False)
        print(f"P2 chooses action {action}")
        state, reward = env.make_move(action, 'p2')
        env.render()


if __name__ == '__main__':
    env = connect_x()

    model = load_model()

    demo(model)