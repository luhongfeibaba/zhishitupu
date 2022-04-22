import tensorflow as tf
import numpy as np


# c = tf.placeholder(tf.float32, [3, None], name='c')
# b = tf.constant(
#     [
#         [1,2,3],
#         [4,5,6],
#         [7,8,9]
#     ],tf.float32
# )
#
# a = tf.matmul(b,c)
# # 代表矩阵b和c相乘赋值给a
# session = tf.Session()
# a1 = session.run(a,feed_dict={c:np.array([[10],[11],[12]])})
# print(a1)
# a2 = session.run(a,feed_dict={c:np.array([[1,2],[3,4],[5,6]])})
# print(a2)
from tensorflow.contrib.framework.python.framework import checkpoint_utils
var_list = checkpoint_utils.list_variables("log/GAN.ckpt")
for v in var_list:
    print(v)