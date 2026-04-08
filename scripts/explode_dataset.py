import random

qas = open("data/qa_finetune/qa_finetune.txt").read().split("\n\n")
big = "\n\n".join(random.sample(qas * 5, len(qas) * 5))
open("data/qa_finetune/qa_big.txt", "w").write(big)