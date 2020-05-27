import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from tensorflow.python import pywrap_tensorflow
# params: pb_file_direction
import argparse


def get_node():
    # function: get the node name of ckpt model
    # checkpoint_path = 'model.ckpt-xxx'
    checkpoint_path = './chinese_L-12_H-768_A-12/bert_model.ckpt'
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
        print("tensor_name: ", key)


def freeze_graph(ckpt, output_graph):
    output_node_names = 'bert/encoder/layer_11/output/dense/kernel'
    saver = tf.train.import_meta_graph(ckpt + '.meta', clear_devices=True)
    # saver = tf.compat.v1.train.import_meta_graph(ckpt + '.meta', clear_devices=True)
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    with tf.Session() as sess:
        saver.restore(sess, ckpt)
        output_graph_def = graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=input_graph_def,
            output_node_names=output_node_names.split(',')
        )
        with tf.gfile.GFile(output_graph, 'wb') as fw:
            fw.write(output_graph_def.SerializeToString())
        print('{} ops in the final graph.'.format(len(output_graph_def.node)))


def print_tensors(pb_file):
    print('Model File: {}\n'.format(pb_file))
    # read pb into graph_def
    with tf.gfile.GFile(pb_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # import graph_def
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)

    # print operations
    for op in graph.get_operations():
        print(op.name + '\t' + str(op.values()))


def print_t():
    model = './bert_model.pb'
    graph = tf.get_default_graph()
    # graph = tf.compat.v1.get_default_graph()
    graph_def = graph.as_graph_def()
    graph_def.ParseFromString(tf.gfile.FastGFile(model, 'rb').read())
    tf.import_graph_def(graph_def, name='graph')
    summaryWriter = tf.summary.FileWriter('log/', graph)
    # summaryWriter = tf.compat.v1.summary.FileWriter('log/', graph)


def show_mode():
    import tensorflow as tf
    from tensorflow.python.platform import gfile

    tf.reset_default_graph()  # 重置计算图
    output_graph_path = 'data/bert_model.pb'
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        output_graph_def = tf.GraphDef()
        # 获得默认的图
        graph = tf.get_default_graph()
        with gfile.FastGFile(output_graph_path, 'rb') as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")
            # 得到当前图有几个操作节点
            print("%d ops in the final graph." % len(output_graph_def.node))

            tensor_name = [tensor.name for tensor in output_graph_def.node]
            print(tensor_name)
            print('---------------------------')
            # 在log_graph文件夹下生产日志文件，可以在tensorboard中可视化模型
            # summaryWriter = tf.summary.FileWriter('log_graph/', graph)

            for op in graph.get_operations():
                # print出tensor的name和值
                print(op.name, op.values())


ckpt = 'dream_output/model.ckpt-19'
pb = 'data/bert_model.pb'

if __name__ == '__main__':
    show_mode()
