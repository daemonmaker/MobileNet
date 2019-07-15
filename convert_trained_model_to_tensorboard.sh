#!/bin/bash

python3 /tensorflow/tensorflow/python/toolsmport_pb_to_tensorboard.py --model_dir mobilenet_v2_1.0_224_quant/mobilenet_v2_1.0_224_quant_frozen.pb --log_dir tboard/
