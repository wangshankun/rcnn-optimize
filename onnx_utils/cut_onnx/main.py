#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import sys
import logging
import argparse
import onnx
import numpy as np
from cut_graph import CutGraph


def main(argv):
    parser = argparse.ArgumentParser(description="Cut onnx")
    parser.add_argument("--src_model", required=True, help="Input Onnx model file")
    parser.add_argument("--dst_model", required=True, help="Cut Out Onnx model file")
    parser.add_argument('--inputs_nodes', required=True, help="input0;input1;intput2")
    parser.add_argument('--outputs_nodes', required=True, help="output0;output1")                     

    args = parser.parse_args(argv)
    #解析input信息

    inputs_info = args.inputs_nodes.split(';')
    outputs_info = args.outputs_nodes.split(';')
    
    model = onnx.load(args.src_model)
    graph = model.graph

    dst_model = CutGraph().cut_in(model, start_n_name=inputs_info, end_n_name=outputs_info)
    dst_model_name = "{}.onnx".format(args.dst_model)
    onnx.save(dst_model, dst_model_name)


if __name__ == "__main__":
    main(sys.argv[1:])
