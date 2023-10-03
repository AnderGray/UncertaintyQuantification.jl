import os
from pathlib import Path
import numpy as np

X1 = {{{ :X1 }}}
X2 = {{{ :X2 }}}

y = X1 + X2

with open('simulation.out', 'w') as f:
    f.write(f"{y}")