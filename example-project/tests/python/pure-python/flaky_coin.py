# retry: 5

import random

print(f"{'heads' if random.random() < 0.1 else 'tails'}")
