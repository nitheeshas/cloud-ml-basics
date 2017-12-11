import numpy as np
import tensorflow as tf

x = tf.placeholder('float', shape=[1, None, 3], name='x')
x_exp = tf.squeeze(x, axis=0)

w = tf.Variable(tf.ones([3, 2]))
y = tf.matmul(x_exp, w)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

tensor_info_x = tf.saved_model.utils.build_tensor_info(x)
print tensor_info_x

tensor_info_y = tf.saved_model.utils.build_tensor_info(y)

prediction_signature = (
    tf.saved_model.signature_def_utils.build_signature_def(
        inputs={'inputs': tensor_info_x},
        outputs={'scores': tensor_info_y},
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
    )

export_path = './test_exports'
builder = tf.saved_model.builder.SavedModelBuilder(export_path)
legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
builder.add_meta_graph_and_variables(
      sess, [tf.saved_model.tag_constants.SERVING],
      signature_def_map={
           'predict_images':
               prediction_signature
      },
      legacy_init_op=legacy_init_op)
builder.save()
