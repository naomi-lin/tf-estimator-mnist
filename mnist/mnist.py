# references:
#   https://github.com/mnoukhov/tf-estimator-mnist
#   https://github.com/tensorflow/models/blob/master/official/mnist/mnist_tpu.py
import tensorflow as tf
import dataset

# pylint: disable=g-bad-import-order
from absl import app as absl_app  # pylint: disable=unused-import

flags = tf.app.flags
# Model specific parameters
flags.DEFINE_string('data_dir', '/tmp/mnist/data',
                    'Directory where mnist data will be downloaded'
                    ' if the data is not already there')
flags.DEFINE_string('model_dir', '/tmp/mnist/model',
                    'Directory where all models are saved')
flags.DEFINE_integer('batch_size', 100,
                     'Mini-batch size for the training. Note that this '
                     'is the global batch size and not the per-shard batch.')
flags.DEFINE_integer('train_steps', 200, 'Total number of training steps.')  # 1000 ---
flags.DEFINE_integer('eval_steps', 100,
                     'Total number of evaluation steps.')
flags.DEFINE_integer('final_eval_steps', 0,
                     'Total number of final evaluation steps. If `0`, evaluation '
                     'after training is skipped.')
flags.DEFINE_float('learning_rate', 0.001, 'Learning Rate')
flags.DEFINE_bool('use_tpu', False, 'Use TPUs rather than plain CPUs')  # ---
# flags.DEFINE_bool('enable_predict', True, 'Do some predictions at the end')
flags.DEFINE_integer('iterations', 50,
                     'Number of iterations per TPU training loop.')
flags.DEFINE_integer('save_checkpoints_steps', 30,
                     'Save checkpoints every this many steps.')
flags.DEFINE_integer('keep_checkpoint_max', 3,
                     'The maximum number of recent checkpoint files to keep.')
flags.DEFINE_integer('save_summary_steps', 30,
                     'Save summaries every this many steps.')


# Cloud TPU Cluster Resolver flags
flags.DEFINE_string('tpu', None,
                    'The Cloud TPU to use for training. This should be either the name '
                    'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 '
                    'url.')
flags.DEFINE_string('tpu_zone', None,
                    '[Optional] GCE zone where the Cloud TPU is located in. If not '
                    'specified, we will attempt to automatically detect the GCE project from '
                    'metadata.')
flags.DEFINE_string('gcp_project', None,
                    '[Optional] Project name for the Cloud TPU-enabled project. If not '
                    'specified, we will attempt to automatically detect the GCE project from '
                    'metadata.')
FLAGS = flags.FLAGS


def metric_fn(labels, logits):
    # use the accuracy as a metric
    accuracy = tf.metrics.accuracy(
        labels=labels,
        predictions=tf.argmax(logits, axis=1))
    return {"accuracy": accuracy}


def train_data(params):
    batch_size = params['batch_size']
    data_dir = params['data_dir']
    data = dataset.train(data_dir)
    data = data.cache().repeat().shuffle(buffer_size=50000)
    data = data.batch(batch_size, drop_remainder=True)  # drop??
    return data


def eval_data(params):
    batch_size = params['batch_size']
    data_dir = params['data_dir']
    data = dataset.train(data_dir)
    # Take out top several samples from test data to make the predictions.
    data = data.cache().repeat().shuffle(buffer_size=50000)  # shuffle too slow ??
    data = data.batch(batch_size, drop_remainder=True)
    return data


def eval_full_data(params):
    batch_size = params['batch_size']
    data_dir = params['data_dir']
    data = dataset.test(data_dir)
    data = data.batch(batch_size, drop_remainder=True)
    return data


def lenet():
    layers = tf.keras.layers

    model = tf.keras.Sequential([
        layers.Reshape(
            target_shape=[28, 28, 1],
            input_shape=(28 * 28,)),

        layers.Conv2D(
            filters=20,
            kernel_size=[5,5],
            padding='same',
            activation=tf.nn.relu),

        layers.MaxPooling2D(
            pool_size=[2,2]),

        layers.Conv2D(
            filters=50,
            kernel_size=[5,5],
            padding='same',
            activation=tf.nn.relu),

        layers.MaxPool2D(
            pool_size=[2,2]),

        layers.Flatten(),

        layers.Dense(
            units=500,
            activation=tf.nn.relu),

        layers.Dense(
            units=10),
    ])

    return model


def model_function(features, labels, mode, params):
    # del params

    # get the model
    model = lenet()

    if mode == tf.estimator.ModeKeys.TRAIN:
        # pass the input through the model
        logits = model(features)

        # selective backprop
        predictions = tf.nn.softmax(logits)  # is this predictions??
        filter_l2 = 50 * tf.sqrt(tf.reduce_sum(tf.square(
            tf.one_hot(labels, tf.shape(logits)[1]) - predictions), 1))
        select_probs = tf.maximum(tf.minimum(filter_l2, 1.0), 0.05)
        pickme = tf.random_uniform([FLAGS.batch_size],
                                   minval=0.0, maxval=1.0)
        chosen = tf.less(pickme, select_probs)
        selected_labels = tf.boolean_mask(labels, chosen)
        selected_logits = tf.boolean_mask(logits, chosen)

        # Not use Selective-Backprop
        # selected_labels = labels[:FLAGS.batch_size/2]
        # selected_logits = logits[:FLAGS.batch_size/2]

        tf.summary.scalar('num_selected_examples', tf.shape(selected_labels)[0])

        # get the cross-entropy loss and name it
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=selected_labels,
            logits=selected_logits)
        tf.identity(loss, 'train_loss')

        # record the accuracy and name it
        accuracy = tf.metrics.accuracy(
            labels=selected_labels,
            predictions=tf.argmax(selected_logits, axis=1))
        tf.identity(accuracy[1], name='train_accuracy')
        tf.summary.scalar('train_accuracy', accuracy[1])

        # use Adam to optimize
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        if FLAGS.use_tpu:
            optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
        tf.identity(FLAGS.learning_rate, name='learning_rate')

        # create an estimator spec to optimize the loss
        estimator_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            # training_hooks=[summary_hook],
            train_op=optimizer.minimize(loss, tf.train.get_global_step()))

    elif mode == tf.estimator.ModeKeys.EVAL:
        # pass the input through the model
        logits = model(features, training=False)

        # get the cross-entropy loss
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=labels,
            logits=logits)

        # create an estimator spec with the loss and accuracy
        estimator_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=loss,
            eval_metrics=(metric_fn, [labels, logits]))

    return estimator_spec


def main(argv):
    del argv

    if FLAGS.use_tpu:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu,
            zone=FLAGS.tpu_zone,
            project=FLAGS.gcp_project
        )

        run_config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            model_dir=FLAGS.model_dir,
            session_config=tf.ConfigProto(
                allow_soft_placement=True, log_device_placement=True),
            tpu_config=tf.contrib.tpu.TPUConfig(FLAGS.iterations),
            save_checkpoints_steps=FLAGS.save_checkpoints_steps,
            keep_checkpoint_max=FLAGS.keep_checkpoint_max,
            save_summary_steps=FLAGS.save_summary_steps
        )
    else:
        # Local
        run_config = tf.contrib.tpu.RunConfig(model_dir=FLAGS.model_dir,
                                              save_checkpoints_steps=FLAGS.save_checkpoints_steps,
                                              keep_checkpoint_max=FLAGS.keep_checkpoint_max,
                                              save_summary_steps=FLAGS.save_summary_steps)

    estimator = tf.contrib.tpu.TPUEstimator(
        model_fn=model_function,
        use_tpu=FLAGS.use_tpu,
        train_batch_size=FLAGS.batch_size,
        eval_batch_size=FLAGS.batch_size,
        predict_batch_size=FLAGS.batch_size,
        params={"data_dir": FLAGS.data_dir},
        config=run_config)

    train_spec = tf.estimator.TrainSpec(input_fn=train_data,
                                        max_steps=FLAGS.train_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_full_data,
                                      start_delay_secs=5,  # start evaluating after N seconds
                                      throttle_secs=5,  # will be less frequent than checkpoint
                                      steps=FLAGS.eval_steps)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    # TPUEstimator.evaluate *requires* a steps argument.
    # Note that the number of examples used during evaluation is
    # --eval_steps * --batch_size.
    # So if you change --batch_size then change --eval_steps too.
    if FLAGS.final_eval_steps:
        estimator.evaluate(input_fn=eval_data, steps=FLAGS.final_eval_steps)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    absl_app.run(main)
