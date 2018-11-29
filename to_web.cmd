cd src
python keras_to_tensorflow.py -input_model_file ..\models\keras\%1 -output_model_file ..\models\tensorflow\%1.pb
tensorflowjs_converter --input_format=tf_frozen_model --output_node_names=output_node0 ..\models\tensorflow\%1.pb ..\models\web\%1\ 
cd ..