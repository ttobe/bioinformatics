from Codes.argparser import argparser
from test import Model
params = argparser()

network = Model(params)

network.train_network()