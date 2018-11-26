import tensorflow as tf
import os
from keras.preprocessing.image import img_to_array, load_img

img = load_img("dataset/test/square_0.png", grayscale=True)
input = img_to_array(img)
input = input.reshape((1,) + input.shape)

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph


model_filename = "./models/tensorflow/model_{}.h5.pb".format(len(next(os.walk("./models/tensorflow/"))[2]) - 1)

# We use our "load_graph" function
graph = load_graph(model_filename)

# We can verify that we can access the list of operations in the graph
for op in graph.get_operations():
    print(op.name)

# We access the input and output nodes
x = graph.get_tensor_by_name('prefix/conv2d_1_input:0')
y = graph.get_tensor_by_name('prefix/output_node0:0')

# We launch a Session
with tf.Session(graph=graph) as sess:
    prediction = sess.run(y, feed_dict={
        x: input
    })

print(
    "Circle : {} \n Hourglass : {} \n Square : {} \n Star : {} \n Triangle : {} \n".format(int(prediction[0][0] * 100),
                                                                                           int(prediction[0][1] * 100),
                                                                                           int(prediction[0][2] * 100),
                                                                                           int(prediction[0][3] * 100),
                                                                                           int(prediction[0][4] * 100)))
