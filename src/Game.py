from enum import Enum


class GameState:
    def __init__(self, dino_position=None, obstacle_position=None, is_over=True):
        # TODO: check if we can get speed information from the frames
        self.game_speed = 0
        self.obstacle_position = obstacle_position
        self.dino_position = dino_position
        self.is_over = is_over



class Action(Enum):
    DO_NOTHING = 0,
    JUMP = 1,
    DUCK = 2