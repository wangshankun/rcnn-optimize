#!/usr/bin/python
# -*- coding: UTF-8 -*-

import logging
import copy
import collections

import onnx
from onnx import ModelProto
from onnx.helper import make_tensor, get_attribute_value
from onnx import helper, shape_inference
from onnx import TensorProto

from .cut_graph import CutGraph

import networkx as nx
from networkx.algorithms import isomorphism

from networkx.drawing.nx_agraph import write_dot, graphviz_layout

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Subgraph(object):
    def __init__(self, sub_digraph):
        self.sub_digraph = sub_digraph
        self._parse_sub()

    def _parse_sub(self):
        self.root_nodes = [n for n, d in self.sub_digraph.in_degree() if d==0]
        self.leaf_nodes = [
            node
            for node, data
            in self.sub_digraph.nodes(data=True)
            if self.sub_digraph.out_degree(node)==0 and self.sub_digraph.in_degree(node)==1
        ]
        self.all_nodes = list(nx.topological_sort(self.sub_digraph))
        self.mid_nodes =list(set(self.all_nodes) - set(self.root_nodes) - set(self.leaf_nodes))


class Transgraph(object):
    def __init__(self, model, hd_op_type, max_sub_num = 3, min_node_num = 10):
        self.model = model
        self.hd_op_type = hd_op_type
        self.max_sub_num = max_sub_num
        self.min_node_num = min_node_num

        self.used_op_type = []
        self.node_dict = {}
        self.init_dict = {}
        self.val_info_dict = {}#存放所有tensor信息value info和input、output
        self.node_by_input = collections.defaultdict(list)
        self.node_by_output = {}

        self._convert_constant_to_init()
        self._check_model()
        self._update_model_property()

        self.input_nodes_is_cpu_set  = set()
        self.output_nodes_is_cpu_set = set()

    def _get_sub_io_node_info(self, sub_t):
        #获取子图输入node
        all_inputs = set()
        init_input = []
        for n_name in sub_t.root_nodes:
            for i in self.node_dict[n_name].input:
                if i not in self.init_dict.keys():
                    all_inputs.add(i)
                else:
                    init_input.append(i)
        #print(all_inputs)
        #获取子图输出node
        all_outputs = set()
        for n_name in sub_t.leaf_nodes:
            n = self.node_dict[n_name]
            all_outputs.update(n.output)
        #print(all_outputs)
        return list(all_inputs), list(all_outputs)

    def exe(self):
        sub_digraphs = self._get_sub_digraphs()
        print("=========valid sub num:%d============"%len(sub_digraphs))
        for sub_idx, sub in enumerate(sub_digraphs):
            #print(list(nx.topological_sort(sub)))
            sub_name = "{}_{}".format("subgraph", str(sub_idx))
            sub_t    = Subgraph(sub)#获取子图节点信息

            #子图权重切割出来，并导出
            sub_model = CutGraph().cut_in(self.model, start_n_name=sub_t.root_nodes, end_n_name=sub_t.leaf_nodes)
            onnx.save(sub_model, "{}.onnx".format(sub_name))
            #print(sub_t.root_nodes)
            #print(sub_t.leaf_nodes)
            all_inputs, all_outputs = self._get_sub_io_node_info(sub_t)
            '''
            #print(self.val_info_dict)
            for node_name in  self.val_info_dict:
                if node_name in all_inputs:
                    print(node_name)
                    for x in self.val_info_dict[node_name].type.tensor_type.shape.dim:
                        print(x.dim_value)
            '''
            #创建新节点
            new_node = onnx.helper.make_node(
                op_type="subgraph",
                inputs=all_inputs,
                outputs=all_outputs,
                name=sub_name
            )
            self.model.graph.node.append(new_node)#新节点插入图中
            #删除旧节点
            useless_nodes = [self.node_dict[i] for i in sub_t.all_nodes]
            self.remove_useless(useless_nodes)

        onnx.save(self.model, "test_fuse.onnx")#保存融合后的模型

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

    def _update_model_property(self):
        # clean
        self.node_dict = {}
        self.init_dict = {}
        self.val_info_dict = {}
        self.node_by_input = collections.defaultdict(list)
        self.node_by_output = {}

        # update
        self.node_dict = {n.name: n for n in self.model.graph.node}
        self.init_dict = {n.name: n for n in self.model.graph.initializer}
        self.val_info_dict.update({n.name: n for n in self.model.graph.value_info})
        self.val_info_dict.update({n.name: n for n in self.model.graph.input})
        self.val_info_dict.update({n.name: n for n in self.model.graph.output})
        for n in self.model.graph.node:
            for i in n.input:
                self.node_by_input[i].append(n)
            for o in n.output:
                self.node_by_output[o] = n

    def _create_digraphs(self):
        self.node_by_input = collections.defaultdict(list)
        for n in self.model.graph.node:
            for i in n.input:
                self.node_by_input[i].append(n)

        self.di_graph = nx.DiGraph()
        nodes = []
        edges = []
        for n in self.model.graph.node:
            # add node
            if n.op_type in self.hd_op_type:#支持npu的优先npu
                nodes.append((n.name, {"op_type": n.op_type, "hd_type": "npu"}))
            else:#默认CPU实现
                nodes.append((n.name, {"op_type": n.op_type, "hd_type": "cpu"}))
            # add edge
            for out in n.output:
                for next_n in self.node_by_input[out]:
                    edges.append((n.name, next_n.name))
        self.di_graph.add_nodes_from(nodes)
        self.di_graph.add_edges_from(edges)

    def _get_sub_digraphs(self):
        #根据model.graph.node创建networkx中的有向图
        self._create_digraphs()

        dig_nodes = self.di_graph.nodes()
        for name, nbrs in self.di_graph.adj.items():#邻接表里面找到邻居(无向图会重复)
            for nbr, _ in nbrs.items():#邻居和链接权重
                #cpu的io相关的节点统计出来
                if dig_nodes[name].get("hd_type") == "cpu":
                    self.input_nodes_is_cpu_set.add(nbr)
                if dig_nodes[nbr].get("hd_type") == "cpu":
                    self.output_nodes_is_cpu_set.add(name)

        #按照支持npu属性的node划分子图
        dig_nodes = [
            node
            for node, data
            in self.di_graph.nodes(data=True)
            if data.get("hd_type") == "npu"
        ]

        self._remove_digraphs_cycle(dig_nodes)
        npu_subgraphs = self.di_no_cycle_graph.subgraph(dig_nodes)
        #过滤符合条件的子图
        valid_sub_digraphs = self._filter_subgraphs(npu_subgraphs)
        return valid_sub_digraphs 

    def _is_all_npu_node(self, nodes):
        ret = True
        for name in nodes:
            if self.di_graph.nodes[name].get("hd_type") == "npu":
                ret = True
            else:
                ret = False
                break
        return ret
        
    def _remove_digraphs_cycle(self, dig_nodes):
        self.di_no_cycle_graph = copy.deepcopy(self.di_graph)

        npu_dig_subgraphs = self.di_graph.subgraph(dig_nodes)
        #先过滤下不符合条件的子图
        filter_sub_digraphs = self._filter_subgraphs(npu_dig_subgraphs)
        #处理子图
        all_need_cut_edges = []
        for sub in filter_sub_digraphs:
            sub_t    = Subgraph(sub)

            hit_in  = set(sub_t.all_nodes) & self.input_nodes_is_cpu_set
            hit_out = set(sub_t.all_nodes) & self.output_nodes_is_cpu_set

            #补全子图(加入CPU的节点，查看是否有环路)
            amend_sub_nodes_name = self._find_node(sub_t.root_nodes, sub_t.leaf_nodes)
            amend_nodes = [
                node
                for node, data
                in self.di_graph.nodes(data=True)
                if node in set(amend_sub_nodes_name)
            ]
            tmp_di_graph = copy.deepcopy(self.di_no_cycle_graph)#切掉所有环路临时需要的图
            amend_dig_sub = tmp_di_graph.subgraph(amend_nodes)
            sub_cycles = nx.cycle_basis(amend_dig_sub.to_undirected())#得到环路
            #拆掉除了CPU通路之外的其它路径
            for cycle in sub_cycles:
                amend_dig_sub = tmp_di_graph.subgraph(amend_nodes)#tmp_di_graph切过一次后再次更新amend_dig_sub
                if len(hit_out & set(cycle)) > 0  and len(hit_in & set(cycle)) > 0:#有NPU和CPU一起参与的环路
                    start_node = list(hit_out & set(cycle))[0]#NPU子图中环的开始节点
                    end_node = list(hit_in & set(cycle))[0]#NPU子图中环的结束节点
                    if start_node == end_node:
                        continue
                    #尝试切断起始节点的所有通路，存储下来切断纯粹NPU路径的的边，反馈给最终的图
                    if nx.has_path(amend_dig_sub.to_undirected(), start_node, end_node):
                        all_paths = list(nx.all_shortest_paths(amend_dig_sub.to_undirected(), start_node, end_node))
                        for path in all_paths:                            
                            edge   = ()
                            edge_r = (path[0], path[1])#选择从起始节点切
                            edge_l = (path[1], path[0])
                            if edge_r in tmp_di_graph.edges:
                                edge = edge_r
                            if edge_l in tmp_di_graph.edges:
                                edge = edge_l

                            if len(edge) == 2:
                                tmp_di_graph.remove_edge(*edge)
                                if self._is_all_npu_node(path):
                                 all_need_cut_edges.append(edge)
        
        #print(all_need_cut_edges)
        #切断所有NPU环路
        for edge in all_need_cut_edges:
            self.di_no_cycle_graph.remove_edge(*edge)
 
    def _filter_subgraphs(self, npu_subgraphs):
        #找到所有独立子图，并且按照子图长度从大到小排序
        npu_subgraphs_sort = sorted(nx.connected_components(npu_subgraphs.to_undirected()), key=len, reverse=True)
        valid_sub_digraphs = []
        for sub in npu_subgraphs_sort:
            g_t = npu_subgraphs.subgraph(sub)
            g_t_node_num = len(list(nx.topological_sort(g_t)))
            if g_t_node_num >= self.min_node_num:
                valid_sub_digraphs.append(g_t)
            if len(valid_sub_digraphs) >= self.max_sub_num:
                break
        return valid_sub_digraphs 

    def _convert_constant_to_init(self):
        '''
        Convert constant node to initializer
        '''
        remove_nodes = [n for n in self.model.graph.node if n.op_type == "Constant"]
        for n in remove_nodes:
            # TODO: sparse_init
            val = get_attribute_value(n.attribute[0])
            val.name = n.output[0]
            self.model.graph.initializer.append(val)
            self.model.graph.node.remove(n)

    def _find_node(self, start_nodes, end_nodes):
        '''
        Find these nodes between start_nodes(included) and end_nodes(included) by recursion.
        :param start_nodes:
        :param end_nodes:
        :return:
        '''
        nodes = []
        def _recursion(n):
            if n.name in end_nodes or n.name in nodes:
                return
            if n.name != "":
                nodes.append(n.name)
            for o in n.output:
                for i in self.node_by_input[o]:
                    _recursion(i)

        for s in start_nodes:
            n = self.node_dict[s]
            _recursion(n)
        nodes.extend(end_nodes)
        return nodes