import os
import sys
import logging
import argparse
import onnx
from subgraph.trans_graph import Transgraph

def main(argv):
    parser = argparse.ArgumentParser(description="ModelTrans && SubgraphFuse.")
    parser.add_argument("--src_model", required=True, help="Onnx model file path.")
    parser.add_argument("--hardware_type", required=True, help="trans to hardware type.")

    args = parser.parse_args(argv)
    model = onnx.load(args.src_model)

    hd_op_type = set()
    for line in open(args.hardware_type): 
        line=line.strip('\n')
        hd_op_type.add(line)
    print(hd_op_type)

    sf = Transgraph(model, hd_op_type)

    sf.show()

if __name__ == "__main__":
    main(sys.argv[1:])
