import tensorrt as trt
logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)
print("TensorRT Runtime created successfully")

