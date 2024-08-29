from easydict import EasyDict as edict
import yaml
class ReadFiles:
    def read_yaml(path):
        try:
            with open(path, 'r') as f:
                file = edict(yaml.load(f, Loader=yaml.FullLoader))
            return file
        except:
            print('NO FILE READ!')
            return None
        
