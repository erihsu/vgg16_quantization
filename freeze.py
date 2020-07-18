import os
import argparse
import tensorflow as tf


def freeze_graph(model_ckpt, output_nodes, rename_nodes=None):
    meta_file = model_ckpt + '.meta'
    if not tf.gfile.Exists(meta_file):
        raise FileNotFoundError('file not found: %s' % meta_file)
    model_dir = "/".join(os.path.abspath(model_ckpt).split('/')[:-1])
    output_model = os.path.join(model_dir, "frozen_model.pb")
    saver = tf.train.import_meta_graph(meta_file, clear_devices=True)

    onames = output_nodes.split(',')
    graph = tf.get_default_graph()
    if rename_nodes is not None:
        rnames = rename_nodes.split(',')
        with graph.as_default():
            for o, n in zip(onames, rnames):
                _out = tf.identity(graph.get_tensor_by_name(o+':0'), name=n)
            onames = rnames

    with tf.Session(graph=graph) as sess:

        saver.restore(sess, model_ckpt)

        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            graph.as_graph_def(),
            onames # unrelated nodes will be discarded
        )

        with tf.gfile.GFile(output_model, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

    return output_graph_def


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_ckpt", type=str, default=None, help="ckpt file to export.")
    parser.add_argument("--output_nodes", type=str, default=None, help="comma separated.")
    parser.add_argument("--rename_nodes", type=str, default=None, help="same order as output_nodes")
    args = parser.parse_args()

    freeze_graph(args.model_ckpt, args.output_nodes, args.rename_nodes)