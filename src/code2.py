import numpy as np

from keras.layers import Input, Dense, Lambda, Concatenate, Reshape
from keras.models import Model
from keras.backend import dot
from keras.engine.topology import Layer
from keras.regularizers import l1


n = 2
m = 3
samples = 100000

b = np.random.randint(low=-1000, high=1000, size=(samples, n, 1))
A = np.random.randint(low=-1000, high=1000, size=(samples, n, m))

print(b.shape, A.shape)


import tensorflow as tf
def my_func(x):
    print(x[0].shape)
    print(x[1].shape)
    d = tf.matmul(x[0], x[1])
    return d

ip_A = Input(shape=(n, m), name='Input_A')
print(A.shape)
ip_A_flat = Reshape(target_shape=(n * m,), name='Input_A_Reshape')(A)
print(ip_A_flat.shape)
ip_b = Input(shape=(n, 1), name='Input_b')
print(b.shape)
ip_b_flat = Reshape(target_shape=(n * 1,), name='Input_b_Reshape')(b)
print(ip_b_flat.shape)
ip = Concatenate(axis=1, name='Input_Fusion')([ip_A_flat, ip_b_flat])
print(ip.shape)

dense = Dense(2000, activation='relu', name='Dense_1')(ip)
print(dense.shape)
dense = Dense(20000, activation='relu', name='Dense_2')(dense)
print(dense.shape)
dense = Dense(2000, activation='relu', name='Dense_3')(dense)
print(dense.shape)
op_x_flat = Dense(m, activation='relu', activity_regularizer=l1(0.1), name='Dense_4')(dense)
print(op_x_flat.shape)
op_x = Reshape(target_shape=(m, 1), name='x_Reshape')(op_x_flat)
print(op_x.shape)

op_Ax = Lambda(my_func)([ip_A, op_x])
#output_Ax = dot(input_A, output_x)
print(op_Ax.shape)

#res = k.matmul(input_A, output_x)

model = Model(inputs=[ip_A, ip_b], outputs=op_Ax)

model.compile(optimizer='adam', loss='mse', metrics=None)

history = model.fit(
    x=[A, b], y=b,
    batch_size=96,
    epochs=100,
    verbose=1,
    callbacks=None,
    shuffle=True,
    initial_epoch=0,
    steps_per_epoch=None,
    validation_data=None)