import tensorflow as tf
import numpy as np
import cv2
import lenet_model
import calib

"""
with tf.Graph().as_default() as g:
    X = tf.placeholder(tf.float32, shape=[None, 784])
    Y = lenet_model.build_model(X)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state('./checkpoint/')
        saver.restore(sess, ckpt.model_checkpoint_path)
        img = cv2.imread('/home/min/code/glow/tests/images/mnist/4_1059.png', 0)
        img = img.reshape(1, 784)

        writer = tf.summary.FileWriter("./log")
        writer.add_graph(sess.graph)
        s, prob = sess.run(Y, feed_dict={X: img})
        writer.add_summary(s, global_step=0)
        print(prob)
"""

"""
ckpt = tf.train.get_checkpoint_state('./checkpoint/')
saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')

matches = quantize._FindLayersToQuantize(tf.get_default_graph())
print([matches[i].layer_op for i in range(len(matches))])

with tf.Session() as sess:
    saver.restore(sess, ckpt.model_checkpoint_path)
    img = cv2.imread('/home/min/code/glow/tests/images/mnist/4_1059.png', 0)
    img = img.reshape(1, 784)
    prob = sess.run(tf.get_default_graph().get_tensor_by_name('fc3/add:0'),
                    feed_dict={'Placeholder:0':img})
    print(prob)
"""

with tf.Graph().as_default():
    graph_def = tf.GraphDef()

    with open('lenet.pb', 'rb') as f:
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')


    with tf.Session() as sess:
        img = cv2.imread('/media/min/newland/code/glow/tests/images/mnist/4_1059.png', 0)
        img = img.reshape(1, 784)
        activations = calib.get_activations(tf.get_default_graph())
        weights = calib.get_weights(tf.get_default_graph())
        calibration_tensors = activations + weights + [tf.get_default_graph().get_tensor_by_name('Placeholder:0')]
        prob = sess.run(calibration_tensors, feed_dict={'Placeholder:0':img})

    min_max=[[np.min(prob[i]), np.max(prob[i])] for i in range(len(prob))]
    tensor_name = [calibration_tensors[i].name for i in range(len(calibration_tensors))]
    quant_info = dict(zip(tensor_name, min_max))
    calib.quantize(tf.get_default_graph(), quant_info)

    f = open('quantized.pb', 'wb')
    f.write(tf.get_default_graph().as_graph_def().SerializeToString())
    f.close()

with tf.Graph().as_default():
    graph_def = tf.GraphDef()

    with open('quantized.pb', 'rb') as f:
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')


    with tf.Session() as sess:
        img = cv2.imread('/media/min/newland/code/glow/tests/images/mnist/4_1059.png', 0)
        img = img.reshape(1, 784)
        prob = sess.run(tf.get_default_graph().get_tensor_by_name('fc3/add/fakequant:0'),
                 feed_dict={'Placeholder:0':img})
        print(prob)
