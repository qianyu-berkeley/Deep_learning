import unittest
import os
import glob
import time
import logging
import numpy as np
import pandas as pd
from shared_lib.misc_utils import Params
from shared_lib import log_utils


json_file = "./config/params.json"


class TestStringMethods(unittest.TestCase):
    def test_Params(self):
        # Read params and set initial value
        p1 = Params(json_file)
        p1.learning_rate = 0.5
        p1.batch_size = 16
        p1.save(json_file)

        # Test read
        p2 = Params(json_file)
        self.assertEqual(p2.learning_rate, 0.5)
        self.assertEqual(p2.batch_size, 16)

        # Update params again
        p2.learning_rate = 0.3
        p2.batch_size = 32
        p2.save(json_file)
        p2.update(json_file)
        self.assertEqual(p2.learning_rate, 0.3)
        self.assertEqual(p2.batch_size, 32)

    def test_logger(self):
        path = "tests"
        filename = "test.log"
        log_utils.set_logger(log_path=os.path.join(path, filename))

        # Test logging coloring
        logging.info("info message")
        logging.warning("warning message")
        logging.error("error message")
        logging.critical("critical message")

        # Test logging file
        self.assertIn(filename, os.listdir(path))
        os.remove(os.path.join(path, filename))


if __name__ == "__main__":
    unittest.main()
