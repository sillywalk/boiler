import os
import sys
import yaml
from pathlib import Path

# Add the main project path to the python path.
root = Path.cwd()
while root.name != "ziegel":
    root = root.parent

proj_dir = root.joinpath("ziegel")
if root not in sys.path:
    sys.path.append(str(proj_dir))

class ConfigBuilder:
    def __init__(self):
        self._num_attributes = num_attributes

    def __iter__(self):
        """
        Iterates over all the attributes.

        Yields
        ------
        (attr_name, attr_val): (str, ConfigBuilder or dict_like)
            Key is the name of the attribute. Value is either an instance of the 
        """
        for attr_name, attr_val in self.__dict__.items():
            yield attr_name, attr_val
    
    @property
    def num_attributes(self):
        """
        A getter method for number of attributes
        """
        return self._num_attributes

    def set_config(self, key, val):
        """
        Set config attribute

        Parameters
        ----------
        key: str
            Name of the config
        val: any
            The value
        """
        # If the config file is nested, then recurse.
        if isinstance(val, dict):
            cfg_cls = ConfigBuilder()
            for sub_key, sub_val in val.items():
                cfg_cls = cfg_cls.set_config(sub_key, sub_val)
            val = cfg_cls

        self._num_attributes += 1
        setattr(self, key, val)
        return self
        
    def load_config(self, config_file):
        """
        Read a yaml file with all the configurations and set them.

        Parameters
        ----------
        config_file: path_str
            Path to the config yaml file

        Returns
        -------
        self: self
            A reference to self
        """
        with open(config_file, 'r') as cfg:
            yaml_loader = yaml.load(cfg, Loader=yaml.FullLoader)
            for attr_name, attr_val in yaml_loader.items():
                self.set_config(attr_name, attr_val)

        return self

    
