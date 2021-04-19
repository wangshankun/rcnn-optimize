#!/usr/bin/python
# -*- coding: UTF-8 -*-

import logging
import copy
import collections

import onnx
from onnx import ModelProto
from onnx.helper import make_tensor, get_attribute_value
import networkx as nx
from networkx.algorithms import isomorphism

from networkx.drawing.nx_agraph import write_dot, graphviz_layout

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Transgraph(object):
    def __init__(self, model, hd_op_type):
        model = self._convert_constant_to_init(model)
        self.hd_op_type = hd_op_type
        self.model = model
        self._check_model()

        self.used_op_type = []
        self.node_dict = {}
        self.init_dict = {}
        # self.node_by_inputs = collections.defaultdict(list)
    def show(self):
        target_digraph = self._get_target_digraph()
        #print(list(nx.topological_sort(target_digraph)))

    def get_tensor_name(self, nodes):
        '''
        :param nodes:
        :return:
        '''
        all_inputs = set()
        all_outputs = set()
        init_input = []
        for n in nodes:
            for i in n.input:
                if i not in self.init_dict.keys():
                    all_inputs.add(i)
                else:
                    init_input.append(i)
            all_outputs.update(n.output)
        inputs = all_inputs - all_outputs
        outputs = all_outputs - all_inputs
        mids = set(all_inputs | all_outputs) - set(inputs) - set(outputs)
        return list(inputs), list(mids), list(outputs), init_input

    def get_input_by_maps(self, obj, match, maps):
        node_name = None
        for k, v in match.items():
            if v == maps[obj][0]:
                node_name = k
        if node_name == None:
            raise Exception("Error.")
        idx = maps[obj][1]
        return self.node_dict[node_name].input[idx]

    def get_output_by_maps(self, obj, match, maps):
        node_name = None
        for k, v in match.items():
            if v == maps[obj][0]:
                node_name = k
        if node_name == None:
            raise Exception("Error.")
        idx = maps[obj][1]
        return self.node_dict[node_name].output[idx]

    def remove_useless(self, nodes):
        '''
        Remove useless nodes, value_info and initializer
        :param nodes:
        :return:
        '''
        # Remove node
        for n in nodes:
            if n in self.model.graph.node:
                self.model.graph.node.remove(n)

        remove_vi = {}
        remove_init = {}
        remove_vi.update({n.name: n for n in self.model.graph.input})
        remove_vi.update({n.name: n for n in self.model.graph.output})
        remove_vi.update({n.name: n for n in self.model.graph.value_info})
        remove_init.update({n.name: n for n in self.model.graph.initializer})

        for n in self.model.graph.node:
            for i in n.input:
                if i in remove_vi.keys():
                    del remove_vi[i]
                if i in remove_init.keys():
                    del remove_init[i]
            for o in n.output:
                if o in remove_vi.keys():
                    del remove_vi[o]
            # search for subgraph, example loop.
            for attr in n.attribute:
                if attr.name in ["body", "else_branch", "then_branch"]:
                    g_t = get_attribute_value(attr)
                    for n_t in g_t.node:
                        for i in n_t.input:
                            if i in remove_vi.keys():
                                del remove_vi[i]
                            if i in remove_init.keys():
                                del remove_init[i]
                        for o in n_t.output:
                            if o in remove_vi.keys():
                                del remove_vi[o]

        # clean value_info
        for k, v in remove_vi.items():
            if v in self.model.graph.input:
                self.model.graph.input.remove(v)
            if v in self.model.graph.output:
                self.model.graph.output.remove(v)
            if v in self.model.graph.value_info:
                self.model.graph.value_info.remove(v)
        # clean init
        for k, v in remove_init.items():
            if v in self.model.graph.initializer:
                self.model.graph.initializer.remove(v)

    def _check_model(self):
        for n in self.model.graph.node:
            if n.name == "":
                raise Exception("This model has node which don`t have name. "
                                "Please use the set_node_name.py in tools to set node name.")

    def _get_model_op_type(self, model):
        '''
        Obtain the all op_type used in model.
        :param model:
        :return:
        '''
        op_type_set = set()
        for n in model.graph.node:
            op_type_set.add(n.op_type)
        return list(op_type_set)

    def _get_target_digraph(self):
        '''
        Convert target onnx model to networkx digraph. Using op_type filter to reduce computation.
        :return: networkx.DiGraph
        '''
        query_model = self._convert_constant_to_init(self.model)
        self.used_op_type = self._get_model_op_type(self.model)
        query_model = ModelProto()
        query_model.graph.node.extend(self.model.graph.node)
        query_model = self._op_type_filter(query_model)
        self._update_model_property()

        return self._create_digraph(query_model)

    def _update_model_property(self):
        self.node_dict.clear()
        self.init_dict.clear()
        self.node_dict = {n.name: n for n in self.model.graph.node}
        self.init_dict = {n.name: n for n in self.model.graph.initializer}

    def _create_digraph(self, model):
        '''
        Create digraph by using onnx model
        :param model:
        :return: networkx.DiGraph
        '''
        node_by_input = collections.defaultdict(list)
        for n in model.graph.node:
            for i in n.input:
                node_by_input[i].append(n)

        di_graph = nx.DiGraph()
        nodes = []
        edges = []
        for n in model.graph.node:
            # add node
            if n.op_type in self.hd_op_type:#支持npu的优先npu
                nodes.append((n.name, {"op_type": n.op_type, "hd_type": "npu"}))
            else:#默认CPU实现
                nodes.append((n.name, {"op_type": n.op_type, "hd_type": "cpu"}))
            # add edge
            for out in n.output:
                for next_n in node_by_input[out]:
                    edges.append((n.name, next_n.name))
        di_graph.add_nodes_from(nodes)
        di_graph.add_edges_from(edges)

        #按照支持npu属性的node划分子图
        nodes = (
            node
            for node, data
            in di_graph.nodes(data=True)
            if data.get("hd_type") == "npu"
        )
        npu_subgraphs = di_graph.subgraph(nodes)
        npu_subgraphs_nodes = list(nx.topological_sort(npu_subgraphs))
        print("===========npu_subgraphs_nodes:===============")
        print(npu_subgraphs_nodes)

        leaf_nodes = [
            node
            for node, data
            in npu_subgraphs.nodes(data=True)
            if npu_subgraphs.out_degree(node)==0 and npu_subgraphs.in_degree(node)==1
        ]

        root_nodes = [n for n, d in npu_subgraphs.in_degree() if d==0]

        print("===========npu_subgraphs root_nodes:===============")
        print(root_nodes)
        print("===========npu_subgraphs leaf_nodes:===============")
        print(leaf_nodes)
        print("===========npu_subgraphs connected_components:===============")
        print(list(nx.connected_components(npu_subgraphs.to_undirected())))

        #找到所有独立子图，并且按照子图长度从大到小排序
        npu_subgraphs_sort = sorted(nx.connected_components(npu_subgraphs.to_undirected()), key=len, reverse=True)
        #找到最大子图
        g_0 = npu_subgraphs.subgraph(npu_subgraphs_sort[0])
        print("===========max subgraph nodes:===============")
        print(list(nx.topological_sort(g_0)))


        return di_graph 

    def _op_type_filter(self, model):
        '''
        :param model:
        :return:
        '''
        remove_node = []
        for n in model.graph.node:
            if n.op_type not in self.used_op_type:
                remove_node.append(n)
        for n in remove_node:
            model.graph.node.remove(n)
        return model

    def _convert_constant_to_init(self, model):
        '''
        Convert constant node to initializer
        :param model:
        :return:
        '''
        remove_nodes = [n for n in model.graph.node if n.op_type == "Constant"]
        for n in remove_nodes:
            # TODO: sparse_init
            val = get_attribute_value(n.attribute[0])
            val.name = n.output[0]
            model.graph.initializer.append(val)
            model.graph.node.remove(n)
        return model
