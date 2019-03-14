import tensorflow as tf


# hello world constant ex
hello = tf.constant('hello world') # immutable

# session defined to execute computation graph
sess = tf.Session()
print(sess.run(hello)) # execute


# variables ex
weights = tf.Variable(tf.random_normal([3,2], stddev=0.1), name='weights') 
a = tf.multiply(2,3) 

tf.global_variables_initializer()

print(sess.run(a))

# placeholders
x = tf.placeholder('float', shape=None)