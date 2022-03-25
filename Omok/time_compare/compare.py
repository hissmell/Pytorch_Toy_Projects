import utils
import time
import random

def measure_time(fn,number=50):
    time_start = time.time()
    for _ in range(number):
        fn()
    print(f"Time : {time.time()-time_start:.6f} sec")

def test_ver1():
    env = utils.Omok()
    env.reset()
    for _ in range(300):
        action = random.choice(env.legal_actions())
        env.step(action)

def test_ver2():
    env = utils.Omok_Ver2()
    env.reset()
    for _ in range(300):
        action = random.choice(env.legal_actions)
        env.step(action)

measure_time(test_ver1)
measure_time(test_ver2)