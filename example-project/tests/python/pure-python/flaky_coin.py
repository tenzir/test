# retry: 5
# runner: python

import random
from pathlib import Path

MAX_ATTEMPTS = 5
STATE = Path(__file__).with_suffix(".state")

attempt = int(STATE.read_text()) + 1 if STATE.exists() else 1

if attempt == 1:
    STATE.write_text(str(attempt))
    print("tails")
    raise SystemExit(1)

if attempt >= MAX_ATTEMPTS or random.random() < 0.25:
    STATE.unlink(missing_ok=True)
    print("heads")
else:
    STATE.write_text(str(attempt))
    print("tails")
    raise SystemExit(1)
