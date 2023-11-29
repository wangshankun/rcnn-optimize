import onnx

# 加载ONNX模型
model = onnx.load('mqbench_qmodel_deploy.onnx')

# 遍历图中的所有节点
for node in model.graph.node:
    # 找到所有的QLinearAdd节点
    if node.op_type == 'QLinearAdd':
        # 获取第四个输入的引用（假设输入的索引从0开始）
        fourth_input = node.input[4]
        # 修改第二个输入为第四个输入的内容
        node.input[1] = fourth_input

# 保存修改后的模型
onnx.save(model, 'mqbench_qmodel_deploy_.onnx')
