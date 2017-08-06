import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from two_different_images_dataset import two_objects_dataset
import tensorflow as tf

sess=tf.InteractiveSession()

dataset_path=r"/media/ahmedsalah/Conda/FCIS/Paths/Machine Learning/Datasets/two different images dataset"
batch_size=10
classes_names=["cat","dog"]
classes_num=len(classes_names)
dataset=two_objects_dataset(batch_size,dataset_path,classes_names)

desired_output = tf.placeholder( tf.float32 , shape=[None,classes_num] )
images_input   = tf.placeholder( tf.float32 , shape=[None,128,128,3] )

conv_layer1_weights = tf.Variable(tf.truncated_normal([3,3,3,32], stddev=0.05)) #fi_sz,fi_sz,inp_chn_num,out_chan_num(filters_num)
conv_layer1_bias    = tf.Variable(tf.constant(.05, shape=[32]))          #bias_num=filters_num
conv_layer1  = tf.nn.conv2d(input=images_input , filter=conv_layer1_weights , strides=[1,1,1,1] , padding='SAME')
conv_layer1 += conv_layer1_bias
conv_layer1  = tf.nn.max_pool(value=conv_layer1 , ksize=[1,2,2,1] , strides=[1,2,2,1] , padding='SAME' )
conv_layer1  = tf.nn.relu(conv_layer1)

conv_layer2_weights = tf.Variable(tf.truncated_normal([3,3,32,32], stddev=0.05))
conv_layer2_bias    = tf.Variable(tf.constant(.05, shape=[32]))
conv_layer2  = tf.nn.conv2d(input=conv_layer1 , filter=conv_layer2_weights , strides=[1,1,1,1] , padding='SAME')
conv_layer2 += conv_layer2_bias
conv_layer2  = tf.nn.max_pool(value=conv_layer2 , ksize=[1,2,2,1] , strides=[1,2,2,1] , padding='SAME' )
conv_layer2  = tf.nn.relu(conv_layer2)

conv_layer3_weights = tf.Variable(tf.truncated_normal([3,3,32,64], stddev=0.05))
conv_layer3_bias    = tf.Variable(tf.constant(.05, shape=[64]))
conv_layer3  = tf.nn.conv2d(input=conv_layer2 , filter=conv_layer3_weights , strides=[1,1,1,1] , padding='SAME')
conv_layer3 += conv_layer3_bias
conv_layer3  = tf.nn.max_pool(value=conv_layer3 , ksize=[1,2,2,1] , strides=[1,2,2,1] , padding='SAME' )
conv_layer3  = tf.nn.relu(conv_layer3)

conv_layer3_shape = conv_layer3.get_shape()
conv_layer3_features_num = conv_layer3_shape[1:4].num_elements()
conv_layer3_flattened = tf.reshape(conv_layer3, [-1, conv_layer3_features_num])

fc_layer1_input_length = conv_layer3_features_num
fc_layer1_neurons_num = 128
fc_layer1_weights = tf.Variable(tf.truncated_normal([conv_layer3_features_num,fc_layer1_neurons_num], stddev=0.05))
fc_layer1_bias = tf.Variable(tf.constant(.05, shape=[fc_layer1_neurons_num]))
fc_layer1 = tf.matmul(conv_layer3_flattened , fc_layer1_weights) + fc_layer1_bias
fc_layer1 = tf.nn.relu(fc_layer1)

fc_output_layer_input_length = fc_layer1_neurons_num
fc_output_layer_neurons_num = 2
fc_output_layer_weights = tf.Variable(tf.truncated_normal([fc_layer1_neurons_num,fc_output_layer_neurons_num], stddev=0.05))
fc_output_layer_bias = tf.Variable(tf.constant(.05, shape=[fc_output_layer_neurons_num]))
fc_output_layer = tf.matmul(fc_layer1 , fc_output_layer_weights) + fc_output_layer_bias


cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=fc_output_layer,labels=desired_output)
cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=.0001).minimize(cost)

#writer = tf.summary.FileWriter("graph folder/",sess.graph)

#sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
saver = tf.train.import_meta_graph('model folder/my_model.meta')
saver.restore(sess, 'model folder/my_model')

for epoch in range(5):
    training_loss=0
    for batch in range(dataset.training_data_size//batch_size):
        training_data, training_labels = dataset.get_next_training_batch()
        feed_dict_training = { images_input:training_data , desired_output:training_labels }
        _,training_cost = sess.run( [optimizer,cost] , feed_dict=feed_dict_training )
        training_loss+=training_cost

    print("Epoch{} training cost: {}".format(epoch+1,training_loss))

    validation_loss=0
    for batch in range(dataset.validation_data_size//batch_size):
        validation_data, validation_labels = dataset.get_next_validation_batch()
        feed_dict_validation = { images_input:validation_data , desired_output:validation_labels }
        validation_cost = sess.run( cost , feed_dict=feed_dict_validation )
        validation_loss+=validation_cost

    print("Epoch{} validation cost: {}".format(epoch + 1, validation_loss))


test_accuracy=0
for batch in range(dataset.test_data_size//batch_size):
    test_data, test_labels = dataset.get_next_test_batch()
    feed_dict_test = { images_input:test_data , desired_output:test_labels }
    logi = tf.argmax(test_labels, 1)
    labe = tf.argmax(fc_output_layer, 1)
    correct_prediction = tf.equal(logi, labe)
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    test_accuracy += sess.run(accuracy, feed_dict=feed_dict_test) * 100
test_accuracy /= dataset.test_data_size//batch_size
print("Test Accuracy : {}%".format(round(test_accuracy,2)))

saver.save(sess, 'model folder/my_model',write_meta_graph=False)


