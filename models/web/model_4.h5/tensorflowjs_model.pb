
8
flatten_1/stack/0Const*
valueB *
dtype0
P
conv2d_1_inputPlaceholder*
dtype0*$
shape:���������
F
conv2d_1/kernelConst*
valueB *
dtype0
8
conv2d_1/biasConst*
value
B *
dtype0
F
conv2d_2/kernelConst*
valueB  *
dtype0
8
conv2d_2/biasConst*
value
B *
dtype0
F
conv2d_3/kernelConst*
valueB @*
dtype0
8
conv2d_3/biasConst*
value
B@*
dtype0
F
conv2d_4/kernelConst*
valueB@@*
dtype0
8
conv2d_4/biasConst*
value
B@*
dtype0
H
flatten_1/strided_slice/stackConst*
value
B*
dtype0
J
flatten_1/strided_slice/stack_1Const*
value
B*
dtype0
?
dense_1/kernelConst*
valueB
��*
dtype0
8
dense_1/biasConst*
valueB	�*
dtype0
>
dense_2/kernelConst*
valueB	�*
dtype0
7
dense_2/biasConst*
value
B*
dtype0
�
conv2d_1/convolutionConv2Dconv2d_1_inputconv2d_1/kernel*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingVALID*
use_cudnn_on_gpu(
`
conv2d_1/BiasAddBiasAddconv2d_1/convolutionconv2d_1/bias*
T0*
data_formatNHWC
4
activation_1/ReluReluconv2d_1/BiasAdd*
T0
�
conv2d_2/convolutionConv2Dactivation_1/Reluconv2d_2/kernel*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingVALID*
use_cudnn_on_gpu(
`
conv2d_2/BiasAddBiasAddconv2d_2/convolutionconv2d_2/bias*
T0*
data_formatNHWC
4
activation_2/ReluReluconv2d_2/BiasAdd*
T0
�
max_pooling2d_1/MaxPoolMaxPoolactivation_2/Relu*
T0*
strides
*
data_formatNHWC*
paddingVALID*
ksize

�
conv2d_3/convolutionConv2Dmax_pooling2d_1/MaxPoolconv2d_3/kernel*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(
`
conv2d_3/BiasAddBiasAddconv2d_3/convolutionconv2d_3/bias*
T0*
data_formatNHWC
4
activation_3/ReluReluconv2d_3/BiasAdd*
T0
�
conv2d_4/convolutionConv2Dactivation_3/Reluconv2d_4/kernel*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingVALID*
use_cudnn_on_gpu(
`
conv2d_4/BiasAddBiasAddconv2d_4/convolutionconv2d_4/bias*
T0*
data_formatNHWC
4
activation_4/ReluReluconv2d_4/BiasAdd*
T0
�
max_pooling2d_2/MaxPoolMaxPoolactivation_4/Relu*
T0*
strides
*
data_formatNHWC*
paddingVALID*
ksize

J
flatten_1/ShapeShapemax_pooling2d_2/MaxPool*
T0*
out_type0
�
flatten_1/strided_sliceStridedSliceflatten_1/Shapeflatten_1/strided_slice/stackflatten_1/strided_slice/stack_1flatten_1/strided_slice/stack*
Index0*
end_mask*
T0*
shrink_axis_mask *
new_axis_mask *

begin_mask *
ellipsis_mask 
v
flatten_1/ProdProdflatten_1/strided_sliceflatten_1/strided_slice/stack_1*
	keep_dims( *
T0*

Tidx0
X
flatten_1/stackPackflatten_1/stack/0flatten_1/Prod*

axis *
T0*
N
]
flatten_1/ReshapeReshapemax_pooling2d_2/MaxPoolflatten_1/stack*
T0*
Tshape0
j
dense_1/MatMulMatMulflatten_1/Reshapedense_1/kernel*
T0*
transpose_b( *
transpose_a( 
X
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/bias*
T0*
data_formatNHWC
3
activation_5/ReluReludense_1/BiasAdd*
T0
j
dense_2/MatMulMatMulactivation_5/Reludense_2/kernel*
T0*
transpose_b( *
transpose_a( 
X
dense_2/BiasAddBiasAdddense_2/MatMuldense_2/bias*
T0*
data_formatNHWC
9
activation_6/SoftmaxSoftmaxdense_2/BiasAdd*
T0
7
output_node0Identityactivation_6/Softmax*
T0 " 