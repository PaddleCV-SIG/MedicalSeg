

import paddle
from paddle import nn
import paddle.nn.functional as F

softmax_helper = lambda x: F.softmax(x, 1)

