import tensorflow as tf

with tf.Session() as sess:
    a = tf.constant(2, name="a")
    b = tf.constant(3, name="b")
    x = tf.add(a, b, name="add")
    writer = tf.summary.FileWriter("graphs", sess.graph)
    print sess.run(x) # >> 5
    writer.close()

