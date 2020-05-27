# encoding: utf-8
# prediction.py
# Tensorflow 1.10.0

import os
import numpy as np
import tensorflow as tf




def model_test(test_data_filename):
    filenames = parse(test_data_filename)
    with tf.gfile.FastGFile('model/my_train.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
    predictions = []
    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('evaluation/out_prob:0')
        i = 0
        for file in filenames:
            image_data = tf.gfile.FastGFile(os.path.join("test_data/", file), 'rb').read()
            prediction = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
            predictions.extend(prediction + 1)
            i = i + 1
            print("第" + str(i) + "张分类完毕")
    return predictions


def main():
    label = model_test("TFcodeX_1.tfrecord")  # 替换为TFcodeX_test.tfrecord
    print("\n预测结果向量：\n", label)


main()
