from enum import Enum

NUMBER_OF_STEPS = 1000
LEARNING_RATE = 0.5
GAMMA = 1  # as we don't want the agent to end the game

class Action(Enum):
    DO_NOTHING = 0,
    JUMP = 1,
    DUCK = 2


class Feature(Enum):
    OBSTACLE_POSITION = 1
    DINO_STATE = 2
    BIRD_POSITION = 3


class GameState:
    def __init__(self, game_speed=0, dist_obstacle=None, dist_bird=None, height=None, is_over=True):
        self.game_speed = game_speed
        self.dist_obstacle = dist_obstacle
        self.dist_bird = dist_bird
        self.bird_height = height
        self.is_over = is_over


class ApproximateQLearningAgent:

    def learn(self):
        initial_weight = 1
        state = GameState(6, None, None, None, False)
        total_reward = 0
        alpha = LEARNING_RATE
        self.weights = {
            Feature.OBSTACLE_POSITION: initial_weight,
            Feature.DINO_STATE: initial_weight,
            Feature.BIRD_POSITION: initial_weight
        }

        for _ in range(NUMBER_OF_STEPS):
            action = self.choose_action(state)
            next_state, reward = self.take_action(action)
            total_reward += reward

            # calculate temporal difference here.
            # update weight of every feature using temporal difference calculated

            state = next_state
            if state.is_over:
                break

    def choose_action(self, state: GameState):
        # implement action selection with optimal q value
        return Action.DO_NOTHING

    def take_action(self, action: Action):
        # take action like JUMP or DUCK or DO_NOTHING
        return GameState(), 0

    def getFeatureWeight(self, feature: Feature):
        pass

    def updateWeight(self, feature: Feature, weight: float):
        pass

    def getFeatureValue(self, feature: Feature, state: GameState, action: Action):
        pass

    #  these methods will hold logic to get feature values from state and action
    def obstacleDistanceFeature(self, state: GameState, action: Action):
        pass

    def dinoStateFeature(self, state: GameState, action: Action):
        pass

    def birdDistanceFeature(self, state: GameState, action: Action):
        pass
