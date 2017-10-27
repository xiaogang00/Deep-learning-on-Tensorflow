import tensorflow as tf      
import tf.contrib.learn   # 
import tf.contrib.slim     


a = tf.add(2, 3)
# Nodes: operators, variables, and constants
# Edges: tensor
sess = tf.Session()
print sess.run(a)
sess.close()     

##To put part of a graph on a specific CPU or GPU:
# Creates a graph.
with tf.device('/gpu:2'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name='b')
    c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print sess.run(c)

#to add operators to a graph, set it as default:
g = tf.Graph()
with g.as_default():
x = tf.add(3, 5)
sess = tf.Session(graph=g)
with tf.Session() as sess:
sess.run(x)
