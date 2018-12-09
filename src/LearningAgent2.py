import datetime
import math
import random
import time

from pyautogui import press, keyDown, keyUp

from src import FindingDistances, ScreenReader
from src.Game import GameState, Action
from src.JsonSerializer import CheckpointHelper

INITIAL_GAME_SPEED = 6
NUMBER_OF_STEPS = 20000
LEARNING_RATE = 0.5
GAMMA = 2  # as we don't want the agent to end the game
GAME_ACCELERATION = 0.003
SCREEN_HEIGHT = 1080
SCREEN_WIDTH = 1920

# TODO: get this height from somewhere
DINO_HEIGHT = 80


def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1.0 + z) + 0.0000000001
    else:
        z = math.exp(x)
        return z / (1.0 + z) + 0.0000000001


def get_dino_y(state: GameState):
    return .9 if state.dino_position is None else state.dino_position[1] / SCREEN_HEIGHT


def get_obs_y(state: GameState):
    return .9 if state.obstacle_position is None else state.obstacle_position[1] / SCREEN_HEIGHT


def get_obs_x(state: GameState):
    return 1 if state.obstacle_position is None else state.obstacle_position[0] / SCREEN_WIDTH


def get_gm_speed(state: GameState):
    return state.game_speed


def get_feature_fn(action: Action, extractor):
    return lambda st, ac: extractor(st) if ac is action else 0


FS1 = {
    Action.JUMP: get_feature_fn(Action.JUMP, get_gm_speed),
    Action.DO_NOTHING: get_feature_fn(Action.DO_NOTHING, get_gm_speed),
    Action.DUCK: get_feature_fn(Action.DUCK, get_gm_speed),
}

FS2 = {
    Action.JUMP: get_feature_fn(Action.JUMP, get_dino_y),
    Action.DO_NOTHING: get_feature_fn(Action.DO_NOTHING, get_dino_y),
    Action.DUCK: get_feature_fn(Action.DUCK, get_dino_y),
}

FS3 = {
    Action.JUMP: get_feature_fn(Action.JUMP, get_obs_y),
    Action.DO_NOTHING: get_feature_fn(Action.DO_NOTHING, get_obs_y),
    Action.DUCK: get_feature_fn(Action.DUCK, get_obs_y),
}

FS4 = {
    Action.JUMP: get_feature_fn(Action.JUMP, get_obs_x),
    Action.DO_NOTHING: get_feature_fn(Action.DO_NOTHING, get_obs_x),
    Action.DUCK: get_feature_fn(Action.DUCK, get_obs_x),
}


class AgentWith12Features:

    def __init__(self):
        self.feature_functions = []
        self.weights = []
        self.sess = None
        self.epsilon = 1
        self.action_set = [Action.DUCK, Action.DO_NOTHING, Action.JUMP]
        self.detector = FindingDistances.Detector()

        date = datetime.datetime.now()
        date_str = str(date.year) + str(date.month) + str(date.day)
        self.ckpt_helper = CheckpointHelper("checkpoints/" + date_str)

    def learn(self):
        # TODO: decide on initial weights
        initial_weight = 0.5
        total_reward = 0
        alpha = LEARNING_RATE
        self.epsilon = 0.9

        self.feature_functions = []
        self.weights = []
        for fs in [FS1, FS2, FS3, FS4]:
            for ac in fs:
                self.feature_functions.append(fs[ac])
                self.weights.append(initial_weight)

        for _ in range(0, NUMBER_OF_STEPS):
            if _ % 30 == 0:
                self.save_weights(_)
            if _ % 100 == 0:
                if self.epsilon > 0.02:
                    self.epsilon = self.epsilon - 0.001

            press('space')
            state = self.get_game_state()
            state.game_speed = INITIAL_GAME_SPEED
            game_speed = INITIAL_GAME_SPEED
            while not state.is_over:
                start_time = time.time()
                if state.dino_position:
                    print("------------------Iteration----------------------")

                    print("state: dino(" + str(state.dino_position) + "), Obs(" + str(state.obstacle_position) + ")")
                    action = self.choose_action(state)
                    print("Take action:" + str(action))
                    next_state, reward = self.take_action(action)
                    next_state.game_speed = game_speed

                    game_speed += GAME_ACCELERATION
                    total_reward += reward

                    # calculate temporal difference here.
                    td_error = (reward + GAMMA * max([self.get_q_value(next_state, act) for act in Action])
                                - self.get_q_value(state, action))
                    print("TD_Error: " + str(td_error))
                    # update weight of every feature using temporal difference calculated
                    for i in range(0, len(self.feature_functions)):
                        feature_fn = self.feature_functions[i]
                        curr_weight = self.weights[i]
                        self.weights[i] = (curr_weight + (alpha * td_error * feature_fn(state, action))) / 100000000007

                    state = next_state

                    print(self.weights)
                else:
                    state = self.get_game_state()
                    state.game_speed = INITIAL_GAME_SPEED
                    game_speed = INITIAL_GAME_SPEED
                print('loop took {} seconds'.format(time.time() - start_time))

    def choose_action(self, state: GameState):
        if random.random() < self.epsilon:
            # select random value
            action_index = random.randrange(len(self.action_set))
            return self.action_set[action_index]
        else:
            # implement action selection with optimal q value
            max_value = float("-Inf")
            max_action = None
            for action in Action:
                value = self.get_q_value(state, action)
                if value > max_value:
                    max_value = value
                    max_action = action
            return max_action

    def get_q_value(self, state, action):
        q_value = 0.0
        for i in range(0, len(self.feature_functions)):
            q_value = self.weights[i] * self.feature_functions[i](state, action)
        return q_value

    def take_action(self, action: Action):
        if action == Action.JUMP:
            self.take_jump()
        elif action == Action.DUCK:
            self.duck()
        game_state = self.get_game_state()

        if game_state.is_over:
            reward = -10
        else:
            reward = 1

        return game_state, reward

    @staticmethod
    def duck():
        keyDown('down')
        time.sleep(0.17)
        keyUp('down')

    @staticmethod
    def take_jump():
        keyDown('space')
        keyUp('space')
        time.sleep(0.32)

    def set_session(self, sess):
        self.sess = sess

    def save_weights(self, ckpt_number):
        self.ckpt_helper.storeCheckpoint(ckpt_number, {'weights': self.weights, 'epsilon': self.epsilon})

    def get_game_state(self):
        return self.detector.run_detection(self.sess, ScreenReader.screen_record_basic())
