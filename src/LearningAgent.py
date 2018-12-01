import math
import time
from enum import Enum

from pyautogui import press, hotkey, keyDown, keyUp

from src import FindingDistances, ScreenReader

NUMBER_OF_STEPS = 1000
LEARNING_RATE = 0.5
GAMMA = 1  # as we don't want the agent to end the game

GAME_ACCELERATION = 0.003

GROUND_Y = 0

# TODO: get this height from somewhere
DINO_HEIGHT = 400


def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1.0 + z) + 0.0000000001
    else:
        z = math.exp(x)
        return z / (1.0 + z) + 0.0000000001


class Action(Enum):
    DO_NOTHING = 0,
    JUMP = 1,
    DUCK = 2


class GameState:
    def __init__(self, dino_position=None, obstacle_position=None, is_over=True):
        # TODO: check if we can get speed information from the frames
        self.game_speed = 0
        self.obstacle_position = obstacle_position
        self.dino_position = dino_position
        self.is_over = is_over


def obstaclePositionFeature(state: GameState, action: Action):
    if not state.obstacle_position:
        if action.DO_NOTHING:
            return 1
        else:
            return 0.5

    dist = state.dino_position[0] - state.obstacle_position[0]
    elevation = abs(state.obstacle_position[1] - GROUND_Y)
    dist_sig = sigmoid(dist)

    # if action == Action.DO_NOTHING:
    #     return dist_sig
    # else:
    #     return 0.5 * dist_sig
    #
    if dist > 50:
        if action is Action.DO_NOTHING:
            return 1 * dist_sig
        else:
            return 0.5 * dist_sig
    else:
        if action is Action.DO_NOTHING:
            return 0.2 * dist_sig
        if action is Action.DUCK:
            return 1 * dist_sig
        if action is Action.JUMP:
            return 1 * dist_sig


def dinoStateFeature(state: GameState, action: Action):
    if state.is_over:
        return -1

    dinoPositionY = state.dino_position[1]
    if abs(GROUND_Y - dinoPositionY) > 10:
        if action is Action.DO_NOTHING:
            return 1
        return .5
    else:
        if action is Action.DO_NOTHING:
            return 1
        return .7


class Feature(Enum):
    OBSTACLE_POSITION = 1, obstaclePositionFeature
    DINO_STATE = 2, dinoStateFeature


class ApproximateQLearningAgent:

    def learn(self):
        # TODO: decide on initial weights
        initial_weight = 0.5
        total_reward = 0
        alpha = LEARNING_RATE
        self.weights = {
            Feature.OBSTACLE_POSITION: initial_weight,
            Feature.DINO_STATE: initial_weight,
        }

        game_speed = 6

        for _ in range(NUMBER_OF_STEPS):
            press('space')
            state = FindingDistances.run_detection(self.sess, ScreenReader.screen_record_basic())
            state.game_speed = 6
            game_speed = 6
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
                    td_error = (reward + GAMMA * max([self.getQValue(next_state, act) for act in Action])
                                - self.getQValue(state, action))
                    print("TD_Error: " + str(td_error))
                    # update weight of every feature using temporal difference calculated
                    for feature in Feature:
                        self.weights[feature] = self.weights[feature] \
                                                + (alpha * td_error * feature.value[1](state, action))

                    state = next_state

                    print(self.weights)
                    print("------------------------------------------------")
                else:
                    state = FindingDistances.run_detection(self.sess, ScreenReader.screen_record_basic())
                    state.game_speed = 6
                    game_speed = 6
                print('loop took {} seconds'.format(time.time() - start_time))

    def choose_action(self, state: GameState):
        # implement action selection with optimal q value
        max_value = float("-Inf")
        max_action = Action.DO_NOTHING
        for action in Action:
            value = self.getQValue(state, action)
            if value > max_value:
                max_value = value
                max_action = action

        return max_action

    def getQValue(self, state, action):
        q_value = 0.0
        for feature in Feature:
            q_value = self.getFeatureWeight(feature) * feature.value[1](state, action)

        return q_value

    def take_action(self, action: Action):
        # take action like JUMP or DUCK or DO_NOTHING
        if action == Action.JUMP:
            keyDown('space')
            time.sleep(0.000001)
            keyUp('space')
        elif action == Action.DUCK:
            press('down')

        gameState = FindingDistances.run_detection(self.sess, ScreenReader.screen_record_basic())

        if gameState.is_over:
            reward = -1
        else:
            reward = .2

        return gameState, reward

    def getFeatureWeight(self, feature: Feature):
        return self.weights[feature]

    def setSession(self, sess):
        self.sess = sess
