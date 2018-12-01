from src import LearningAgent, FindingDistances, ScreenReader
from src.LearningAgent import ApproximateQLearningAgent
import tensorflow as tf
import cv2 as cv
import numpy as np


def start_playing():
    graph_def = FindingDistances.setup_detection_environment()

    with tf.Session() as sess:
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
        count = 0

        img = cv.imread('static_game.png')
        img = np.array(img)
        gameState = FindingDistances.run_detection(sess, img)
        LearningAgent.GROUND_Y = gameState.dino_position[1]

        agent = ApproximateQLearningAgent()
        agent.setSession(sess)

        ScreenReader.open_chrome()
        agent.learn()


if __name__ == '__main__':
    start_playing()
