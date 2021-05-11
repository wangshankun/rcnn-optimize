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
    def __init__(self, model, tensor_format, npu_op_types, max_sub_num = 3, min_node_num = 10):
        self.model = model
        self.tensor_format = tensor_format
        self.npu_op_types = npu_op_types
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

    def exe(self):
        sub_digraphs = self._get_sub_digraphs()
        print("=========valid sub num:%d============"%len(sub_digraphs))
        for sub_idx, sub in enumerate(sub_digraphs):
            #print(list(nx.topological_sort(sub)))
            sub_name = "{}_{}".format("subgraph", str(sub_idx))
            sub_inputs  = []
            sub_outputs = []
            sub_all_nodes = list(nx.topological_sort(sub))
            for node in sub_all_nodes:
                if node in self.sub_input_nodes:
                    sub_inputs.append(node)
                if node in self.sub_output_nodes:
                    sub_outputs.append(node)                    

            #子图权重切割出来，并导出
            sub_model = CutGraph().cut_in(self.model, start_n_name=sub_inputs, end_n_name=sub_outputs)
            sub_model_name = "{}.onnx".format(sub_name)
            onnx.save(sub_model, sub_model_name)
            sub_onnx_graph = sub_model.graph
            #根据input信息设置模型输入graph shape
            inputs_ = []      
            outputs_ = []
            for input_tensor in sub_onnx_graph.input:
                if input_tensor.name not in self.init_dict.keys():
                    inputs_.append(input_tensor.name)

            for output_tensor in sub_onnx_graph.output:
                outputs_.append(output_tensor.name)

            #'''
            #print(self.val_info_dict)
            inputs_info_   = ''
            outputs_info_ = ''
            for node_name in  self.val_info_dict:
                if node_name in inputs_:
                    format_in = '%s:float:%s:%s;'
                    dims = []
                    for x in self.val_info_dict[node_name].type.tensor_type.shape.dim:
                        dims.append(x.dim_value)
                    inputs_info_ = inputs_info_ + format_in%(node_name, self.tensor_format, ','.join(str(x) for x in dims))

                if node_name in outputs_:
                    format_out = '%s:float:%s:%s;'
                    dims = []
                    for x in self.val_info_dict[node_name].type.tensor_type.shape.dim:
                         dims.append(x.dim_value)
                    outputs_info_ = outputs_info_ + format_out%(node_name, self.tensor_format, ','.join(str(x) for x in dims))
            #'''
            #创建新节点
            new_node = onnx.helper.make_node(
                op_type="SubGraph",
                inputs=inputs_,
                outputs=outputs_,
                name=sub_name,
                inputs_info=inputs_info_,
                outputs_info=outputs_info_
            )
            self.model.graph.node.append(new_node)#新节点插入图中
            #删除旧节点
            useless_nodes = [self.node_dict[i] for i in sub_all_nodes]
            self.remove_useless(useless_nodes)

        onnx.save(self.model, "fuse_graph.onnx")#保存融合后的模型

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
        self.nodes_io_info = {}
        self.sub_input_nodes = []#切割后的所有的子图中应该为input的节点名
        self.sub_output_nodes = []#切割后的所有的子图中应该为output的节点名

        # update
        self.node_dict = {n.name: n for n in self.model.graph.node}
        self.init_dict = {n.name: n for n in self.model.graph.initializer}
        self.val_info_dict.update({n.name: n for n in self.model.graph.value_info})
        self.val_info_dict.update({n.name: n for n in self.model.graph.input})
        self.val_info_dict.update({n.name: n for n in self.model.graph.output})
        for n in self.model.graph.node:
            self.nodes_io_info[n.name]={'input':[],'output':[]}
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
            if n.op_type in self.npu_op_types:#支持npu的优先npu
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
                self.nodes_io_info[name]['output'].append(nbr)
                self.nodes_io_info[nbr]['input'].append(name)

        #按照支持npu属性的node划分子图
        dig_nodes = [
            node
            for node, data
            in self.di_graph.nodes(data=True)
            if data.get("hd_type") == "npu"
        ]

        valid_sub_digraphs = self._remove_digraphs_cycle(dig_nodes)
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
    
    def _topological_sort(self, nodes):
            di_topo_nodes = list(nx.topological_sort(self.di_graph))
            sort_nodes = []
            for x in di_topo_nodes:
                if x in nodes:
                    sort_nodes.append(x)
            return sort_nodes

    def _remove_digraphs_cycle(self, dig_nodes):
        npu_dig_subgraphs = self.di_graph.subgraph(dig_nodes)
        #先过滤下不符合条件的子图
        filter_sub_digraphs = self._filter_subgraphs(npu_dig_subgraphs)
        #处理子图(可能存在于子图之外的环路，需要继续切割子图)
        tmp_di_graph = copy.deepcopy(self.di_graph)#切掉子图所有环路临时需要的整图
        remove_edges = []
        for sub in filter_sub_digraphs:
            #print(list(nx.topological_sort(sub)))
            sub_t    = Subgraph(sub)
            #子图的根、叶子节点按照整网拓扑排序
            sub_t.root_nodes = self._topological_sort(sub_t.root_nodes)
            sub_t.leaf_nodes = self._topological_sort(sub_t.leaf_nodes)
            #print(sub_t.root_nodes)
            #print(sub_t.leaf_nodes)
            for leaf in sub_t.leaf_nodes:
                for root in sub_t.root_nodes:#切除环路过程:
                    while nx.has_path(tmp_di_graph, leaf, root):#如果npu子图的叶子到根有路,证明子网通过CPU有路,会成环;
                        #print("%s to % has_path"%(leaf, root))
                        #这里采用无环图的目的是为了从叶子节点出发找出回路中在整图拓扑逻辑中第一个节点
                        all_paths = list(nx.all_shortest_paths(tmp_di_graph.to_undirected(), leaf, root))
                        for path in all_paths:#
                            topo_1_leaf = self._topological_sort(path)[0]#根据整图逻辑排序，出现在路径中的第一个节点
                            top_down_paths = list(nx.all_shortest_paths(tmp_di_graph.to_undirected(), topo_1_leaf, root))
                            for td_path in top_down_paths:
                                for idx, node in enumerate(td_path):
                                    if node == leaf:#如果是等于输入的叶子节点，那么直接断！
                                        edge = (td_path[0], td_path[1])
                                        if edge in tmp_di_graph.edges:
                                            tmp_di_graph.remove_edge(*edge)
                                            remove_edges.append(edge)
                                        break
                                    #从底向上有向图找，遇到第一个通的节点就是要打断的节点,然后break出来
                                    if (nx.has_path(tmp_di_graph, root, node)):#如果存在那么idx一定大于零
                                        edge = (td_path[idx - 1], td_path[idx])
                                        if edge in tmp_di_graph.edges:
                                            tmp_di_graph.remove_edge(*edge)
                                            remove_edges.append(edge)
                                        break

            #切除子图叶子与output的边; 上面切除环路的循环逻辑中也会切掉部分叶子节点(node == leaf)情况
            for leaf in sub_t.leaf_nodes:
                for i in self.nodes_io_info[leaf]['output']:
                    edge   = (leaf, i)
                    if edge in tmp_di_graph.edges:
                        tmp_di_graph.remove_edge(*edge)
                        remove_edges.append(edge)

            #切除子图根与input的边；
            for root in sub_t.root_nodes:
                for i in self.nodes_io_info[root]['input']:
                    edge   = (i, root)
                    if edge in tmp_di_graph.edges:
                        tmp_di_graph.remove_edge(*edge)
                        remove_edges.append(edge)

        #print(remove_edges)
        for edge in remove_edges:
            self.sub_input_nodes.append(edge[1])#被别人断掉的节点，是本图的输入节点
            self.sub_output_nodes.append(edge[0])#主动断开别人路的节点，是上个图的输出节点

        npu_subs = []
        subs = self._filter_subgraphs(tmp_di_graph)#找出所有处理后的子图
        for sub in subs:
            #print(len(subs))
            #print(list(nx.topological_sort(sub)))
            if self._is_all_npu_node(list(nx.topological_sort(sub))):#只返回全部NPU节点的子图
                npu_subs.append(sub)
        return npu_subs

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