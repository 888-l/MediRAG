''' Testing '''

import evaluation
import os

model_path = "./runs/runX/checkpoint/NTN_relation/model_best.pth.tar"
# model_path = "./runs/runX/checkpoint/NTN_relation/model_best.pth.tar"
data_path = "./data/"
top = evaluation.evalrank(model_path, data_path=data_path, split="test")
