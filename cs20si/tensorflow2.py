import tensorflow as tf     

tf.linspace(10.0, 13.0, 4) ## ==> [10.0 11.0 12.0 13.0]
# 'start' is 3, 'limit' is 18, 'delta' is 3
tf.range(start, limit, delta) ## ==> [3, 6, 9, 12, 15]
# 'limit' is 5
tf.range(limit) ## ==> [0, 1, 2, 3, 4]
# but Tensor objects are not iterable, not for "for"


# input_tensor is [0, 1], [2, 3], [4, 5]]
tf.zeros_like(input_tensor) ## ==> [[0, 0], [0, 0], [0, 0]]

tf.ones(shape, dtype=tf.float32, name=None)
tf.ones_like(input_tensor, dtype=None, name=None, optimize=True)

