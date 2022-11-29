import yaml

class Config(object):
    def __init__(self, config):
        stream = open(config,'r')
        docs = yaml.load_all(stream, Loader=yaml.FullLoader)
        for doc in docs:
            for k, v in doc.items():
                cmd = "self." + k + "=" + repr(v)
                print(cmd)
                exec(cmd)

        stream.close()
