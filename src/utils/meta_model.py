from abc import ABCMeta, abstractmethod

class MetaModel(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, config):
        pass

    @abstractmethod
    def _configure_inputs_and_outputs(self):
        """
        An abstract method to read (and perhaps pre-process) the input data. Then, create placeholders for inputs and outputs.
        """
        pass

    @abstractmethod
    def _construct_neuron_layer(self, layers_config):
        """
        An abstract method to construct a neuron layer. Upon inheritence, one
        may define their own construction of the neuron layer, or choose to use
        one of Tensorflow's many inbuilt neuron layer constructors. 

        Parameters
        ----------
        layers_config: utils.configurator.ConfigBuilder
            All pertinent information required to build a neuron layer.
        
        Returns
        -------
        z: 1-D tensor
            The output activations. 
        """
        pass
    
    @abstractmethod
    def _constuct_deep_neural_network(self):
        """
        An abstract method to construct a deep neural network.

        Returns
        -------
        output: 1-D tensor
            The outputs from the final layer of the DNN. 
        """
        pass

    @abstractmethod
    def _compute_loss(self):
        """
        An abstract method to compute the loss at any given state. It is possible to use user defined loss, or use one of many inbuilt loss functions.

        Returns
        -------
        loss: 1-D tensor
            The loss value
        """
    
    @abstractmethod
    def _evaluate_outcomes(self):
        """
        An abstract method to evaluate the predictions. 

        Notes
        -----
        The method signature for this is not strictly enforced. Use best judgement to define an appropriate method signature based on application.
        """

