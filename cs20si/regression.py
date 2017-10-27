import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd
DATA_FILE="data/fire_theft.xls"
# Step 1: read in data from the .xls file
book = xlrd.open_workbook(DATA_FILE, encoding_override ​= ​"utf-8")
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
# 在这里是读取一列的数据
n_samples = sheet.nrows - 1
# Step 2: create placeholders for input X (number of fire) and label Y (number of theft)
X = tf.placeholder(tf.float32, name = "X")
Y = tf.placeholder(tf.float32, name = "Y")
# Step 3: create weight and bias, initialized to 0
w = tf.Variable(0.0, name = "weights")
b = tf.Variable(0.0, name = "bias")
# Step 4: construct model to predict Y (number of theft) from the number of fire
Y_predicted = X * w + b
# Step 5: use the square error as the loss function
loss = tf.square(Y - Y_predicted, name = "loss")
# Step 6: using gradient descent with learning rate of 0.01 to minimize loss
optimizer ​= ​ tf.train.​GradientDescentOptimizer(learning_rate=0.001).minimize ​(loss)
with tf.Session() as sess:
    # Step 7: initialize the necessary variables, in this case, w and b
    sess.run(tf.global_variables_initializer())
    # Step 8: train the model
    for i in range ​(100): ​ # run 100 epochs
       for x, y ​in data: # Session runs train_op to minimize loss
           sess.run(optimizer, feed_dict ​={X:x ​, Y:y})
    w_value, b_value ​= sess.run([w, b])

