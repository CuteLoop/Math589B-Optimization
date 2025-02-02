# utils.py
import logging
import time
from pathlib import Path

def setup_logger(name, log_file, level=logging.INFO, formatter=None):
    """
    Sets up and returns a logger with a given name and log file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Avoid duplicate logging

    # Create a file handler if one doesn't exist already.
    if not logger.handlers:
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        if formatter is None:
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

def time_function(func, *args, **kwargs):
    """
    Times the execution of a function using time.perf_counter.
    
    Returns:
        result: The result from the function call.
        elapsed: The elapsed time in seconds.
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed

def log_runtime(logger, implementation_name, elapsed_time, extra=""):
    """
    Logs the total runtime information.
    """
    logger.info(f"[{implementation_name}] Total runtime: {elapsed_time:.4f} seconds. {extra}")

# Example usage when run as a script
if __name__ == '__main__':
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)

    # Set up a simple logger for runtime tracking
    runtime_logger = setup_logger("runtime", "logs/runtime.log", level=logging.INFO)
    
    # Define a dummy function to test the timing
    def dummy_function(n):
        total = 0
        for i in range(n):
            total += i
        return total

    # Time the dummy function and log the runtime

    for i in range(10):
        result, elapsed = time_function(dummy_function, 10**6)
        log_runtime(runtime_logger, "DummyFunction", elapsed, extra=f"Result: {result}")
