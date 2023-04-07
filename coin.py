import random

def coin(p1):
    r=random.random()
    if r<(1-p1):
        return 0
    else:
        return 1