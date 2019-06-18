import numpy as np
import tensorflow as tf

class Vanilla:
    def __init__(self, config):
        self.config = config
    
    @staticmethod
    def _construct_neuron_layer(prev_layer, n_neurons, scope, activation=None):
        """
        Construct a fully connected of layer of DNN.

        Parameters
        ----------
        prev_layer: array_like
            The previous layer consisting of n_inputs. The first layer is the  
            original input from training samples.
        n_neurons: int
            The number of neurons in the current layer. 
        scope: str
            Used to label the current name space. 
        activation: str
            The activation function to be used at the outputs.
        
        Returns
        -------
        z: array_like
            The output activations. 
        """

        with tf.name_scope(scope):
            input_size = int(prev_layer.get_shape()[1])
            stddev = 2 / np.sqrt(input_size)
            W_init = tf.truncated_normal(stddev=stddev)
            b_init = tf.random_uniform([n_neurons], minval=-1.0, maxval=1.0, seed=1729)
            W = tf.Variable(W_init(shape=(input_size, n_neurons)), name="weights")
            b = tf.Variable(b_init, name="biases")
            z = tf.matmul(X, W) + b
            if activation.casefold() == 'relu'.casefold():
                z = tf.nn.relu(z)
            elif activation.casefold() == 'tanh'.casefold():
                z = tf.nn.tanh(z)
            elif activation.casefold() == 'sigmoid'.casefold():
                z = tf.nn.sigmoid(z)    
            return z
    
    def _construct_dnn(self, scope):
        """
        Constructs 
        """
        with tf.name_scope(scope):
            for layer_name, layer_info in self.config["layers"].items():
                if layer_name == "input":
                    prev_layer = layer_info["layer_input"]
                else:
                    prev_layer = output

                n_neurons = layer_info["layer_size"]
                activation = layer_info["activation"]
                output = self._construct_neuron_layer(prev_layer, n_neurons, layer_name, activation)
            
            return output
    
    def _train_dnn(self, loss_fn=None):
        """
        """
        
        if loss_fn in None:
            with tf.name_scope("loss"):
                xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)


    


