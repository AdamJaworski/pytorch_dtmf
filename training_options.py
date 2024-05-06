class Options:
    def __init__(self):
        pass


opt = Options()


def add_argument(**kwargs):
    for key, value in kwargs.items():
        setattr(opt, key, value)


add_argument(SAVE_MODEL_AFTER=1000)
add_argument(PRINT_RESULTS=500)
add_argument(CONTINUE_LEARNING=True)
add_argument(MODEL='Gamma')
add_argument(LR=1e-4)
add_argument(LR_DROPOFF_FACTOR=0.5)
add_argument(STARTING_NUMBER=2950000)

