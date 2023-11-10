import onnx
from onnx import helper, TensorProto

'''
org_model_name = "ssd-r34-onnxqnn_4w8a_tensor_deploy_deploy.onnx"
cut_model_name = "s_tensor_4w8a_mix.onnx"
tmp_model_name = "ssd-r34-onnxqnn_4w8a_tensor_deploy_deploy_tmp.onnx"
'''

org_model_name = "ssd-r34-as-onnxqnn_4w4a_channel_mix_deploy_deploy.onnx"
cut_model_name = "as_channel_48layer_mix_res.onnx"
tmp_model_name = "ssd-r34-as-onnxqnn_4w4a_channel_mix_deploy_deploy_tmp.onnx"

# 载入原始模型
model = onnx.load(org_model_name)

# 获取模型的第一个输入的名称和数据类型
input_tensor = model.graph.input[0]
input_name = input_tensor.name
input_data_type = input_tensor.type.tensor_type.elem_type

# 更新模型的输入维度，将批量大小从4更改为1
new_input_shape = [1, 3, 320, 320]  # 新的输入形状

# 为模型的输入创建一个新的 TypeProto 对象
new_input_type_proto = helper.make_tensor_type_proto(input_data_type, new_input_shape)

# 设置新的维度
input_tensor.type.CopyFrom(new_input_type_proto)


output_tensor = helper.make_tensor_value_info(
    name='1096',  # 输出层名称
    elem_type=TensorProto.INT8,  # 数据类型
    shape=[1, 64, 80, 80]  # 形状
)

# 将新的输出信息添加到模型的图定义中
model.graph.output.append(output_tensor)

# 保存修改后的模型
onnx.save(model, tmp_model_name)

# 现在，您可以尝试使用修改后的模型来提取部分模型
input_names = ["1082_QuantizeLinear"]
output_names = ["1096"]
onnx.utils.extract_model(tmp_model_name, cut_model_name, input_names, output_names)



# 重新载入修改后的模型
modified_model = onnx.load(cut_model_name)
# 验证模型
onnx.checker.check_model(modified_model)
