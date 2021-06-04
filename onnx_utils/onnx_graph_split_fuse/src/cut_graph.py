import collections
import onnx
from onnx import ModelProto

class CutGraph:
    def __init__(self):
        self.model = None
        self.node_dict = {}
        self.init_dict = {}
        self.val_info_dict = {}
        self.node_by_input = collections.defaultdict(list)
        self.node_by_output = {}

    def cut_in(self, model, start_n_name=[], end_n_name=[]):
        '''
        Reserve these nodes between start and end.
        :param model:
        :param start_n_name:
        :param end_n_name:
        :return:
        '''
        self._update_model(model)

        if "" in start_n_name or "" in end_n_name:
            raise Exception("Your start_n_name or end_n_name have empty string. "
                            "It's illegal")

        # If start_node is None, set start_node input nodes
        # If end_nodes is None, set end_node output nodes
        if start_n_name == []:
            for m_i in self.model.graph.input:
                if m_i.name not in self.init_dict.keys():
                    for n in self.node_by_input[m_i.name]:
                        start_n_name.append(n.name)
        if end_n_name == []:
            for m_o in self.model.graph.output:
                end_n_name.append(self.node_by_output[m_o.name].name)

        nodes_name = self._find_node(start_n_name, end_n_name)
        inputs, mids, outputs, init_input = self._get_tensor(nodes_name)

        # TODO topo

        inputs.extend(init_input)
        subgraph_model = ModelProto()
        subgraph_model.graph.input.extend([self.val_info_dict[n] for n in inputs if n in self.val_info_dict.keys()])
        subgraph_model.graph.output.extend([self.val_info_dict[n] for n in outputs if n in self.val_info_dict.keys()])
        subgraph_model.graph.node.extend([self.node_dict[n] for n in nodes_name if n in self.node_dict.keys()])
        subgraph_model.graph.initializer.extend([self.init_dict[n] for n in init_input if n in self.init_dict.keys()])
        subgraph_model.graph.value_info.extend([self.val_info_dict[n] for n in mids if n in self.val_info_dict.keys()])

        subgraph_model.ir_version = self.model.ir_version
        subgraph_model.opset_import.extend(self.model.opset_import)
        return subgraph_model

    def cut_out(self, model, start_n_name=[], end_n_name=[]):
        """
        Remove these nodes between start and end.
        :param model:
        :param start_n_name:
        :param end_n_name:
        :return:
        """
        self._update_model(model)

        # If start_node is None, set start_node input nodes
        # If end_nodes is None, set end_node output nodes
        if start_n_name == []:
            for m_i in self.model.graph.input:
                if m_i.name not in self.init_dict.keys():
                    for n in self.node_by_input[m_i.name]:
                        start_n_name.append(n.name)
        if end_n_name == []:
            for m_o in self.model.graph.output:
                end_n_name.append(self.node_by_output[m_o.name].name)

        nodes_name = self._find_node(start_n_name, end_n_name)
        inputs, mids, outputs, init_input = self._get_tensor(nodes_name)
        nodes = [self.node_dict[n] for n in nodes_name]
        self._clean_useless(nodes, inputs, mids, outputs, init_input)

        # TODO: add lose input or output
        return self.model

    def _clean_useless(self, nodes, inputs, mids, outputs, init_input):
        # clean node
        for n in nodes:
            if n in self.model.graph.node:
                self.model.graph.node.remove(n)

        remove_vi = []
        remove_vi.extend(inputs)
        remove_vi.extend(mids)
        remove_vi.extend(outputs)
        remove_vi.extend(init_input)
        for n in self.model.graph.node:
            for i in n.input:
                if i in remove_vi:
                    remove_vi.remove(i)
            for o in n.output:
                if o in remove_vi:
                    remove_vi.remove(o)

        for vi in remove_vi:
            # Clean val_info
            vi_t = self.val_info_dict.get(vi)
            if vi_t != None:
                if vi_t in self.model.graph.input:
                    self.model.graph.input.remove(vi_t)
                if vi_t in self.model.graph.output:
                    self.model.graph.output.remove(vi_t)
                if vi_t in self.model.graph.value_info:
                    self.model.graph.value_info.remove(vi_t)

            # Clean init
            init_t = self.init_dict.get(vi)
            if init_t != None and init_t in self.model.graph.initializer:
                self.model.graph.initializer.remove(init_t)

    def _update_model(self, model):
        self.model = model
        self._check_model()

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
        self.node_by_input = collections.defaultdict(list)
        self.node_by_output = {}

        for n in self.model.graph.node:
            for i in n.input:
                self.node_by_input[i].append(n)
            for o in n.output:
                self.node_by_output[o] = n

    def _check_model(self):
        '''
        Check model each node which has name, if someone dont`t have name, rasise err.
        :return:
        '''
        # check node has a name
        for n in self.model.graph.node:
            if n.name == "":
                raise Exception("This model has node which don`t have name. "
                                "Please use the set_node_name.py in tools to set node name.")

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

    def _get_tensor(self, nodes_name):
        '''
        Get all tensor in these nodes, including input, output, mid and init
        :param nodes_name:
        :return:
        '''
        all_inputs = set()
        all_outputs = set()
        init_input = []

        for n in nodes_name:
            n_t = self.node_dict[n]
            for i in n_t.input:
                if i not in self.init_dict.keys():
                    all_inputs.add(i)
                else:
                    init_input.append(i)
            all_outputs.update(n_t.output)

        inputs = all_inputs - all_outputs
        outputs = all_outputs - all_inputs
        mids = set(all_inputs | all_outputs) - set(inputs) - set(outputs)

        return list(inputs), list(mids), list(outputs), init_input

    def _topo_dfs(self):
        pass

    def _topo_bfs(self):
        pass


if __name__ == "__main__":
    pass
