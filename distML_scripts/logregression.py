import numpy as np
import tensorflow as tf
import os
import json

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('lr', 0.00003, 'Initial learning rate')
tf.app.flags.DEFINE_integer('steps_to_validate', 1000, 'Step to validate and print loss')

learning_rate = FLAGS.lr
steps_to_validate = FLAGS.steps_to_validate

def main(_):
    tf_config = json.loads(os.environ["TF_CONFIG"])
    job_name = tf_config["task"]["type"]
    task_index = tf_config["task"]["index"]
    ps_hosts = tf_config["cluster"]["ps"]
    print("ps hosts are: ", ps_hosts)
    worker_hosts = tf_config["cluster"]["worker"]
    print("worker hosts are: ", worker_hosts)
    master_hosts = tf_config["cluster"]["master"]
    print("master hosts are: ", master_hosts)
    is_sync = os.environ["IS_SYNC"]
    if is_sync == 'True' or is_sync == True:
	tf.app.flags.DEFINE_integer("issync", 1, "Whether to adopt Distributed Synchronization Mode, 1: sync, 0:async")
	print('SYNC MODE SET TO TRUE !!!!!')
    elif is_sync == 'False' or is_sync == False:
	tf.app.flags.DEFINE_integer("issync", 0, "Whether to adopt Distributed Synchronization Mode, 1: sync, 0:async")
	print('SYNC MODE SET TO FALSE !!!!!')

    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker":worker_hosts, "master":master_hosts})
    server = tf.train.Server(cluster, job_name=job_name, task_index=task_index)

    issync = FLAGS.issync
    if job_name == "ps":
        server.join()
    elif job_name == "worker" or job_name == "master":
        work_dev= ""
        if job_name == "worker":
            work_dev = "/job:worker/task:%d" % task_index
        elif job_name == "master":
            work_dev = "/job:master/task:%d" % task_index

        with tf.device(tf.train.replica_device_setter(worker_device = work_dev, cluster=cluster)):
            global_step = tf.Variable(0, name='global_step', trainable=False)

            input = tf.placeholder("float", [None,1])
            label = tf.placeholder("float", [None,1])

            weight = tf.get_variable("weight", [1,1], tf.float32, initializer=tf.random_normal_initializer())
            biase = tf.get_variable("biase", [1,1], tf.float32, initializer=tf.random_normal_initializer())
            pred = tf.multiply(input, weight) + biase

            loss_value = loss(label, pred)
            #loss_value = tf.reshape(loss_value)
            weight_summary = tf.reshape(weight,[1,1])
            biase_summary = tf.reshape(biase,[1,1])
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)

            grads_and_vars = optimizer.compute_gradients(loss_value)

            if issync == 1:
                # Update gradients in Synchronization Mode
                rep_op = tf.train.SyncReplicasOptimizer(optimizer, 
                                                        replicas_to_aggregate=len(worker_hosts) + len(master_hosts),
                                                        total_num_replicas=len(worker_hosts) + len(master_hosts),
                                                        use_locking=True
                                                        )

                train_op = rep_op.apply_gradients(grads_and_vars,
                                                  global_step=global_step
                                                 )

                init_token_op = rep_op.get_init_tokens_op()
                chief_queue_runner = rep_op.get_chief_queue_runner()

            else:
                # Update gradients in Asynchronization Mode
                train_op = optimizer.apply_gradients(grads_and_vars,
                                                     global_step=global_step
                                                    )

            ischief = False

	    saver = tf.train.Saver()
            tf.summary.tensor_summary('Weight', weight_summary)
            tf.summary.tensor_summary('Biase', biase_summary)

            tf.summary.tensor_summary('cost', loss_value)
            summary_op = tf.summary.merge_all()
	    #batch_size = 128
	    #total_size=128*50000
	    #arr = np.zeros([total_size, 2], dtype='float')
	    #for i in range(0,total_size):
		#x = np.random.randn(1)
		#arr[i,0] = x
		#arr[i,1] = 2 * x + np.random.randn(1) * 0.33 + 10

	    #dataset = tf.data.Dataset.from_tensor_slices(arr)
	    #dataset = dataset.batch(batch_size=batch_size)
	    #x_input,y_label = dataset.make_one_shot_iterator().get_next()

	    init_op = tf.global_variables_initializer()
            if job_name == "master" and task_index==0:
                ischief = True

	    log_dir = "/root"
            sv = tf.train.Supervisor(is_chief=ischief,
                                     logdir=log_dir,
                                     init_op=init_op,
                                     summary_op=None, #summary_op,
                                     saver=saver,
                                     global_step=global_step,
                                     save_model_secs=60)

            with sv.prepare_or_wait_for_session(server.target) as sess:
                # If is Synchronization Mode
                if job_name == "master" and task_index == 0 and issync == 1:
                    sv.start_queue_runners(sess, [chief_queue_runner])
                    sess.run(init_token_op)
                step = 0
		batch_size=128
                while step < 50000:
		    xarr=np.zeros((batch_size, 1))
		    yarr=np.zeros((batch_size, 1))
		    for i in range(0,batch_size):
		        x = np.random.randn(1)
			xarr[i]=x
			yarr[i]=2 * x + np.random.randn(1) * 0.33 + 10
                    _, loss_v, step, summary = sess.run([train_op, loss_value, global_step, summary_op], feed_dict={input:xarr, label:yarr})
		    w, b = sess.run([weight, biase])
		    if step % steps_to_validate == 0:
			#print("step: %d, weight: %f, bias: %f, loss: %f" %(step, w, b, loss_v))
			print("processed 1000 steps for every mini-batch!!")
                    if ischief and step % steps_to_validate == 0:
                        sv.summary_computed(sess, summary)

            sv.stop()

def loss(label, pred):
    return tf.square(label - pred)

if __name__ == "__main__":
    tf.app.run()
