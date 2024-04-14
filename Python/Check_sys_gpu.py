import sys
import argparse
import tensorflow.keras
import pandas as pd
import sklearn as sk
import scipy as sp
import tensorflow as tf
import platform
import datasets

def check_system_and_libraries():
    """
    Function to check and print the versions of the system and key libraries used in data science and machine learning.
    It also tests TensorFlow operations to check if GPU acceleration is available.

    Example:
    python Check_sys_gpu.py
    """
    print(f"Python Platform: {platform.platform()}")
    print(f"Tensor Flow Version: {tf.__version__}")
    print(f"Keras Version: {tensorflow.keras.__version__}")
    print()
    print(f"Python {sys.version}")
    print(f"Pandas {pd.__version__}")
    print(f"Scikit-Learn {sk.__version__}")
    print(f"SciPy {sp.__version__}")
    
    gpu = len(tf.config.list_physical_devices('GPU')) > 0
    print("GPU is", "available" if gpu else "NOT AVAILABLE")

    tf.debugging.set_log_device_placement(True)
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)
    print(c)
    print("If GPU is used, you should see the work is running on device:GPU:0")

def main():
    parser = argparse.ArgumentParser(description="Check the system and library versions for ML development.")
    # You can add command line options here if needed in the future
    args = parser.parse_args()
    
    check_system_and_libraries()

if __name__ == "__main__":
    main()
