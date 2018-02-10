class Config(object):
    def __init__(self):
        self.MAX_SENTENCE_LENGTH = 500
        self.MAX_WORD_LENGTH = 4
        self.COMP_WIDTH = 3
        self.CHAR_EMB_DIM = 15
        self.BATCH_SIZE = 100
        self.WORD_DIM = 600
        self.MAX_RUN = 1
        self.VERBOSE = 0
        self.EPOCHS = 50