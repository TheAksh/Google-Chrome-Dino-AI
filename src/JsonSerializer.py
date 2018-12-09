import io
import json
import os


class CheckpointHelper:
    def __init__(self, folder_name):
        self.folder_name = folder_name
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

    def storeCheckpoint(self, ckptNum, data):
        with io.open(self.folder_name+'/chkpt-' + str(ckptNum) + '.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
