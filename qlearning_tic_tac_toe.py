import random

BLANK = ''
AI_PLAYER = 'X'
HUMAN_PLAYER = 'O'
TRAINING_EPOCHS = 40000
TRAINING_EPSILON = .4

REWARD_WIN = 10
REWARD_LOSE = -10
REWARD_TIE = -3


class Player:

    @staticmethod
    def show_board(self, board):
        print('|'.join(board[0:3]))
        print('|'.join(board[3:6]))
        print('|'.join(board[6:9]))


class HumanPlayer(Player):
    def reward(self, value, board):
        pass

    def make_move(self, board):
        while True:
            try:
                self.show_board(board)
                move = input('Your next move cell index 1-9: ')
                move = int(move)

                if not (move - 1 in range(9)):
                    raise ValueError
            except ValueError:
                print('Invalid move; try again:\n')
            else:
                return move - 1


class AIPlayer(Player):
    def __init__(self, epsilon=0.4, alpha=0.3, gamma=0.9, default_q=1):
        # this is the epsilon parameter of th emodel: the probability of exploration
        self.EPSILON = epsilon
        # learning rate
        self.ALPHA = alpha
        # discount parameter for future reward (rewards now are better than rewards in the future)
        self.GAMMA = gamma
        self.DEFAULT_Q = default_q
        # Q(s,a) function is a dict in this implementation.
        self.q = {}
        # prev. move in the game
        self.move = None
        self.board = (' ',) * 9

    def available_moves(self, board):
        return [i for i in range(9) if board[i] == ' ']

    def get_q(self, state, action):
        if self.q.get((state, action)) is None:
            self.q[(state, action)] = self.DEFAULT_Q
        return self.q[(state, action)]

    # make a random move with eps probability (exploration) or pick the action with the highest Q value
    # (exploitation)
    def make_move(self, board):
        self.board = tuple(board)
        actions = self.available_moves(board)

        # action with eps probability
        if random.random() < self.EPSILON:
            self.move = random.choice(actions)
            return self.move

        # take the actino with the highest Q value
        q_values = [self.get_q(self, a) for a in actions]
        max_q_value = max(q_values)

        # if there are multiple best actions, chose one at random
        if q_values.count(max_q_value) > 1:
            best_actions = [i for i in range(len(actions)) if q_values[i] == max_q_value]
            best_move = actions[random.choice(best_actions)]
        else:
            best_move = actions[q_values.index(max_q_value)]

        self.move = best_move

        return self.move

    def reward(self, reward, board):
        if self.move:
            prev_q = self.get_q(self.board, self.move)
            max_q_new = max([self.get_q(tuple(board), a) for a in self.available_moves(self.board)])
            self.q[(self.board, self.move)] = prev_q + self.ALPHA * (reward + self.GAMMA * max_q_new - prev_q)


class TicTacToe:
    def __init__(self, player1, player2):
        self.player1 = player1
        self.player2 = player2
        self.first_player_turn = random.choice([True, False])
        self.board = [' '] * 9

    def play(self):
        while True:
            if self.first_player_turn:
                player = self.player1
                other_player = self.player2
                player_tickers = (AI_PLAYER, HUMAN_PLAYER)
            else:
                player = self.player2
                other_player = self.player1
                player_tickers = (HUMAN_PLAYER, AI_PLAYER)

            # check the state of the game
            game_over, winner = self.is_game_over(player_tickers)

            # game is over: handle rewards
            if game_over:
                if winner == player_tickers[0]:
                    player.show_board(self.board[:])
                    print('\n %s won' % player.__class__.__name__)
                    player.reward(REWARD_WIN, self.board[:])
                    other_player.reward(REWARD_LOSE, self.board[:])
            if winner == player_tickers[1]:
                player.show_board(self.board[:])
                print('\n %s won' % player.__class__.__name__)
                other_player.reward(REWARD_WIN, self.bord[:])
                player.reward(REWARD_LOSE, self.board[:])
            else:
                player.show_board(self.board[:])
                print('Tie!')
                player.reward(REWARD_TIE, self.board[:])
                other_player.reward(REWARD_TIE, self.board[:])
                break

            self.first_player_turn = not self.first_player_turn

            move = player.make_move(self.board)
            self.board[move] = player_tickers[0]
           