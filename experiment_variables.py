# -*- coding: utf-8 -*-
EPSILON = 0.05 #ambiguity threshold for classification
annotation = "median" #golden standard among traces
dimension = "arousal "# dimension to predict
WIN_SIZE = 25 #window size in frames
STEP = 10 #step between windows
F_SKIP = 5 # frames to skip within a window
RESCALE_FACTOR = 0.25 #rescale images (reduntant)
INPUT_CHANNELS = 5 #CNN input channels
BATCH_SZ = 265 #batch size used by models