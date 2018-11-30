from src.object_detection import FindingDistances
from src.screen_reader import ScreenReader


def start_playing():
    ScreenReader.open_chrome_2()
    FindingDistances.setup_detection_environment()
    FindingDistances.run_detection(ScreenReader.screen_record_basic())

if __name__ == '__main__':
    start_playing()
