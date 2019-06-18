import os
import sys
import numpy
import unittest
from pathlib import Path
from pdb import set_trace
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Add the main project path to the python path.
root = Path.cwd()
while root.name != "ziegel":
    root = root.parent
test_dir = root.joinpath("tests")
proj_dir = root.joinpath("ziegel")
if root not in sys.path:
    sys.path.append(str(proj_dir))

from utils.configurator import ConfigBuilder

class TestBasic(unittest.TestCase):
    def test_configurator(self):
        test_yaml = test_dir.joinpath("test_files", "test_yaml_file.yml")
        config_builder = ConfigBuilder()
        config = config_builder.load_config(config_file=test_yaml)
        self.assertEqual(config._num_attributes, 4)
        self.assertEqual(config.n_layers, 2)
        self.assertEqual(config.layers._num_attributes, 2)
        self.assertEqual(config.output.n_neurons, 10)
        
if __name__ == '__main__':
    unittest.main()
