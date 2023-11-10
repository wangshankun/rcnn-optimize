
import onnx
from onnx import helper, TensorProto


# 定义输入和输出信息
input_tensor = helper.make_tensor_value_info('1451', TensorProto.UINT8, [1, 2048, 7, 7])
output_tensor = helper.make_tensor_value_info('1453', TensorProto.UINT8, [1, 2048, 1, 1])

# 创建量化参数的 TensorProto（如果它们是常量）
# 请根据您的模型调整 scale 和 zero_point 的值
input_scale_tensor = helper.make_tensor('layer4_2_relu_2_post_act_fake_quantizer.scale', TensorProto.FLOAT, [1], [1.5315335988998413])  # 示例 scale 值
input_zero_point_tensor = helper.make_tensor('layer4_2_relu_2_post_act_fake_quantizer.zero_point', TensorProto.INT8, [1], [0])  # 示例 zero_point 值
output_scale_tensor = helper.make_tensor('flatten_post_act_fake_quantizer.scale', TensorProto.FLOAT, [1], [0.03122865967452526])  # 示例 scale 值
output_zero_point_tensor = helper.make_tensor('flatten_post_act_fake_quantizer.zero_point', TensorProto.INT8, [1], [0])  # 示例 zero_point 值


# 创建算子节点
node = onnx.helper.make_node(
    'QLinearGlobalAveragePool',
    inputs=[
        '1451',
        'layer4_2_relu_2_post_act_fake_quantizer.scale',
        'layer4_2_relu_2_post_act_fake_quantizer.zero_point',
        'flatten_post_act_fake_quantizer.scale',
        'flatten_post_act_fake_quantizer.zero_point'
    ],
    outputs=['1453'],
    domain='com.microsoft'  # 指定自定义域
)

# 创建图
graph = helper.make_graph(
    nodes=[node],
    name='quantized-model-graph',
    inputs=[input_tensor],  # 输入仅包括图的主要输入
    outputs=[output_tensor],
    initializer=[
        input_scale_tensor, 
        input_zero_point_tensor, 
        output_scale_tensor, 
        output_zero_point_tensor
    ]  # 初始化器列表包括所有的常量
)

# 创建模型
model = helper.make_model(
    graph,
    producer_name='pytorch',
    opset_imports=[
        helper.make_opsetid('', 14),
        helper.make_opsetid('com.microsoft', 1),
    ]
)

# 设置模型属性
model.ir_version = onnx.IR_VERSION
model.model_version = 6
model.producer_version = '1.10'

# 保存模型
onnx.save(model, 'quantized_model.onnx')

# 验证模型
onnx.checker.check_model(model)

'''

ONNX 运行时需要明确的输入和输出信息来执行模型。对于您提供的算子，它需要有定义好的输入和输出张量。如果这些信息已经存在，ONNX 运行时应该能够执行它。如果没有，或者您需要为这个算子创建一个新的独立模型，您需要按照以下步骤进行.


您提供的模型属性信息显示它是用 ONNX v6 格式创建的，由 PyTorch 1.10 生成，并且导入了 com.microsoft 域的 v1 和 ai.onnx 域的 v14。这意味着模型可能使用了 Microsoft 的自定义算子，并且需要特定版本的 ONNX 运行时才能执行。
要在 ONNX 模型中指定这些信息，您可以在创建模型时设置模型的域和运算符集版本。


在这个例子中，inputs 和 outputs 属性在创建节点时被填充，以包含量化参数的名称。这些量化参数必须与模型中定义的名称相匹配。同时，您还需要创建这些量化参数的 ValueInfoProto 对象，并将它们添加到图的输入和输出列表中。
请注意，您可能需要使用 make_tensor 来为量化参数创建初始化器（如果它们不是动态的），并将这些初始化器添加到图的 initializer 列表中。


'''
