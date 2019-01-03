from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
from tensorflow.contrib.quantize.python import common
from tensorflow.contrib.quantize.python import graph_matcher
from tensorflow.contrib.quantize.python import input_to_ops
from tensorflow.python.framework import ops

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope

# Quantizable operation types that are supported by the calibration rewrite.
_QUANTIZABLE_TYPES = {'Conv2D', 'MatMul', 'DepthwiseConv2dNative'}

# Activations that are supported by the quantization rewrite.
_ACTIVATION_TYPES = {'Relu', 'Relu6', 'Identity'}

_RELU_TYPES = {'Relu', 'Relu6'}

_QUANTIZATION_OP = {'FakeQuantWithMinMaxVars'}
_VALID_SRC_OP = {'Add', 'Mul'}
_INTERMEDIATE_OP = {'Add', 'Mul'}
_PASS_THROUGH_OP = {'Reshape', 'Identity', 'BatchToSpaceND', 'SpaceToBatchND'}
_VALID_ACTIVATION_OP = {'Relu', 'Relu6'}

def get_activations(graph):
  """Get activations to be calibrated.

  Args:
    graph: Graph to be checked.
  Raises:
    ValueError: When check fails.
  """
  layer_matches = _FindLayersToCalibration(graph)
  activations = []
  for layer_match in layer_matches:
    for output in layer_match.activation_op.outputs:
      activations.append(output)

  return activations

def get_weights(graph):
  """Get weight tensors to be calibrated.

  Args:
    graph: Graph to be calibrated.
  Raises:
    ValueError: When check fails.
  """
  layer_matches = _FindLayersToCalibration(graph)
  weights = []
  for layer_match in layer_matches:
    weights.append(layer_match.weight_tensor)

  return weights


def quantize(graph, quantize_info):
  """Quantize the graph with quantize_info.

  Args:
    graph: Graph to be modified.
    quantize_info: Quantization info in dictionary format.
  Raises:
    ValueError: When quantization fails.
  """
  for tensor_name, min_max in quantize_info.items():
    tensor = graph.get_tensor_by_name(tensor_name)
    name = tensor_name.split(':')[0]
    consumers = tensor.consumers()
    quant = array_ops.fake_quant_with_min_max_args(
              tensor,
              min=min_max[0],
              max=min_max[1],
              name=name+'/fakequant'
              )

    if consumers:
      modified_count = common.RerouteTensor(
              quant, tensor, can_modify=consumers)
      # Some operations can have multiple output tensors going to the same
      # consumer. Since consumers is a set, we need to ensure that
      # modified_count is greater than or equal to the length of the set
      # of consumers.
      if modified_count < len(consumers):
        raise ValueError('No inputs quantized for ops: [%s]' % ', '.join(
              [consumer.name for consumer in consumers]))

def ModifyForCalibration(graph,
                         vars_collection=ops.GraphKeys.GLOBAL_VARIABLES,
                         scope=None):
  """Update graph with calibration operations.

  Args:
    graph: Graph to modify.
    scope: The scope to be transformed. If it's not None, only the ops which
      are in this scope will be transformed
  Raises:
    ValueError:When modification fails.
  """
  if scope and not scope.endswith('/'):
    scope += '/'

  input_to_ops_map = input_to_ops.InputToOps(graph)
  calib_ops = []

  for layer_match in _FindLayersToCalibration(graph):
    # Quantize the weights.
    context = _GetContextFromOp(layer_match.layer_op)

    # If `scope` is given, only quantize it if the consumer of weights
    # (the layer op) is in the right scope.
    if layer_match.weight_tensor is not None:
      print(layer_match.weight_tensor.op)
      calib_op = _InsertCalibOp(
                  context,
                  'weights_calib',
                  layer_match.weight_tensor.op,
                  input_to_ops_map.ConsumerOperations(layer_match.weight_tensor.op),
                  vars_collection=vars_collection,
                  consumer_scope=scope)
      calib_ops.append(calib_op)

  return calib_ops

def _FindLayersToCalibration(graph):
  """Matches layers in graph to quantize.
  The following patterns get matched. Nodes surrounded by [] will be
  optionally matched:
          weight|folded_weight
                /
         conv|fc
            |
      [batch_to_space_nd]
            |
    [post_conv_correction]
            |
     [biasadd|folded_bias]
            |
         [bypass]
            |
        activation
            |
   [post_activation_bypass]
  Match replacements:
    If weight|folded_weight is found, FakeQuant is added afterwards.
    If bypass is found, FakeQuant is added before and after.
    If activation is found, FakeQuant is added afterwards.
    If post_activation_bypass is found, FakeQuant is added afterwards.
  Args:
    graph: Graph to perform match on.
  Returns:
    list of _LayerMatches.
  """
  input_pattern = graph_matcher.OpTypePattern('*')
  weight_var_pattern = graph_matcher.OpTypePattern('Variable|VariableV2')
  weight_const_pattern = graph_matcher.OpTypePattern('Const')
  weight_partition_identity_pattern = graph_matcher.OpTypePattern(
      'Identity', inputs=[weight_var_pattern, weight_const_pattern])
  weight_partition_concat_pattern = graph_matcher.OpTypePattern(
      'ConcatV2', inputs=[weight_partition_identity_pattern, '*', '*'])
  weight_identity_pattern = graph_matcher.OpTypePattern(
      'Identity',
      inputs=[
          graph_matcher.OneofPattern([
              weight_partition_identity_pattern,
              weight_partition_concat_pattern,
              weight_var_pattern,
              weight_const_pattern
          ])
      ])
  weight_resource_var_pattern = graph_matcher.OpTypePattern('ReadVariableOp')
  folded_weight_pattern = graph_matcher.OpTypePattern('Mul')

  # The weights inputs to the layer operation can either be from the Variable or
  # the folded weight (Mul).
  layer_pattern = graph_matcher.OpTypePattern(
      '|'.join(_QUANTIZABLE_TYPES),
      inputs=[
          input_pattern,
          graph_matcher.OneofPattern([
              weight_identity_pattern, weight_resource_var_pattern,
              folded_weight_pattern
          ])
      ],
      ordered_inputs=False)

  # For atrous convolutions a BatchToSpaceND will occur after the depthwise
  # convolution.
  batch_to_space_pattern = graph_matcher.OpTypePattern(
      'BatchToSpaceND',
      inputs=[
          layer_pattern,
          graph_matcher.OpTypePattern('*'),
          graph_matcher.OpTypePattern('*')
      ])

  layer_output_pattern = graph_matcher.OneofPattern(
      [batch_to_space_pattern, layer_pattern])

  # For separable convolutions, we are looking for a conv, followed by a conv
  # with no activations between the two.
  sep_conv_pattern = graph_matcher.OpTypePattern(
      '|'.join(_QUANTIZABLE_TYPES),
      inputs=[
          graph_matcher.OneofPattern([layer_output_pattern]),
          graph_matcher.OpTypePattern('*')
      ],
      ordered_inputs=False)
  folded_bias_mul_pattern = graph_matcher.OpTypePattern(
      'Mul',
      inputs=[graph_matcher.OpTypePattern('*'), layer_output_pattern],
      ordered_inputs=False)
  post_layer_op_correction_pattern = graph_matcher.OpTypePattern(
      'Add',
      inputs=[folded_bias_mul_pattern,
              graph_matcher.OpTypePattern('*')],
      ordered_inputs=False)
  folded_bias_add_pattern = graph_matcher.OpTypePattern(
      'Add',
      inputs=[
          post_layer_op_correction_pattern,
          graph_matcher.OpTypePattern('*')
      ],
      ordered_inputs=False)

  # batch_norms with forced updates have an Identity operation at the end.
  # TODO(suharshs): Find a way to easily skip extra Identity operations. The
  # current issue is that doing so can often match patterns across many layers
  # incorrectly.
  batch_norm_identity = graph_matcher.OpTypePattern(
      'Identity', inputs=[folded_bias_add_pattern])

  bias_add_pattern = graph_matcher.OpTypePattern(
      'Add|BiasAdd', inputs=[layer_output_pattern, '*'], ordered_inputs=False)

  # The bias can come from the bias add or the folded bias add.
  bypass_pattern = graph_matcher.OpTypePattern(
      'Add',
      inputs=[
          graph_matcher.OneofPattern(
              [bias_add_pattern, folded_bias_add_pattern, batch_norm_identity]),
          '*'
      ],
      ordered_inputs=False)

  # The input to the activation can come from bias add, fold bias add, the
  # bypasses.
  # TODO(suharshs): We should ideally skip Identity operations instead of
  # treating them as activations.
  activation_pattern = graph_matcher.OpTypePattern(
      '|'.join(_ACTIVATION_TYPES) + '|Identity',
      inputs=[
          graph_matcher.OneofPattern([
              bias_add_pattern,
              folded_bias_add_pattern,
              batch_norm_identity,
              bypass_pattern,
              layer_pattern,
          ])
      ])

  post_activation_bypass_pattern = graph_matcher.OpTypePattern(
      'Add', inputs=['*', activation_pattern], ordered_inputs=False)

  # The order of the following matching blocks is very important. Since matches
  # aren't guaranteed to be disjoint, we structure matches from largest to
  # smallest to guarantee that the largest match always wins. Additionally, we
  # ensure that we don't match layers multiple times.

  layer_matches = []
  # We use matched_layer_set to ensure that layers aren't matched multiple
  # times.
  matched_layer_set = set()

  # First, we match layers that have a post activation bypass. We do this first
  # to ensure we don't match only the first part of this layer, missing the
  # post activation bypass node.
  post_activation_bypass_layer_matcher = graph_matcher.GraphMatcher(
      post_activation_bypass_pattern)
  for match_result in post_activation_bypass_layer_matcher.match_graph(graph):
    layer_op = match_result.get_op(layer_pattern)
    weight_tensor = match_result.get_tensor(weight_identity_pattern)
    if weight_tensor is None:
      weight_tensor = match_result.get_tensor(weight_resource_var_pattern)
    if weight_tensor is None:
      weight_tensor = match_result.get_tensor(folded_weight_pattern)
    activation_op = match_result.get_op(activation_pattern)
    bias_add_op = match_result.get_op(bias_add_pattern)
    if bias_add_op is None:
      bias_add_op = match_result.get_op(folded_bias_add_pattern)
    bypass_op = match_result.get_op(bypass_pattern)
    post_activation_bypass_op = match_result.get_op(
        post_activation_bypass_pattern)
    if layer_op not in matched_layer_set:
      matched_layer_set.add(layer_op)
      layer_matches.append(
          _LayerMatch(layer_op, weight_tensor, activation_op, bypass_op,
                      post_activation_bypass_op, bias_add_op))

  # Now, we match the basic layer ending at an activation. We may get duplicate
  # matches from above, but we don't add them to layer_matches.
  layer_matcher = graph_matcher.GraphMatcher(activation_pattern)
  for match_result in layer_matcher.match_graph(graph):
    layer_op = match_result.get_op(layer_pattern)
    weight_tensor = match_result.get_tensor(weight_identity_pattern)
    if weight_tensor is None:
      weight_tensor = match_result.get_tensor(weight_resource_var_pattern)
    if weight_tensor is None:
      weight_tensor = match_result.get_tensor(folded_weight_pattern)
    activation_op = match_result.get_op(activation_pattern)
    bias_add_op = match_result.get_op(bias_add_pattern)
    if bias_add_op is None:
      bias_add_op = match_result.get_op(folded_bias_add_pattern)
    bypass_op = match_result.get_op(bypass_pattern)
    if layer_op not in matched_layer_set:
      if not _IsSkipLayer(activation_op):
        matched_layer_set.add(layer_op)
        layer_matches.append(
            _LayerMatch(layer_op, weight_tensor, activation_op, bypass_op, None,
                        bias_add_op))

  # Match the final layer, where there may not be an activation and instead
  # the output of the final BiasAdd must be quantized. So we treat the BiasAdd
  # as the 'activation_op' in the _LayerMatch, to ensure that it's output is
  # quantized.
  final_layer_matcher = graph_matcher.GraphMatcher(
      graph_matcher.OneofPattern([bias_add_pattern, folded_bias_add_pattern]))
  for match_result in final_layer_matcher.match_graph(graph):
    layer_op = match_result.get_op(layer_pattern)
    weight_tensor = match_result.get_tensor(weight_identity_pattern)
    if weight_tensor is None:
      weight_tensor = match_result.get_tensor(weight_resource_var_pattern)
    if weight_tensor is None:
      weight_tensor = match_result.get_tensor(folded_weight_pattern)
    activation_op = match_result.get_op(bias_add_pattern)
    if activation_op is None:
      activation_op = match_result.get_op(folded_bias_add_pattern)
    if layer_op not in matched_layer_set:
      matched_layer_set.add(layer_op)
      layer_matches.append(
          _LayerMatch(layer_op, weight_tensor, activation_op, None, None, None))

  # Look for separable convolutions here
  sep_conv_matcher = graph_matcher.GraphMatcher(sep_conv_pattern)
  for match_result in sep_conv_matcher.match_graph(graph):
    layer_op = match_result.get_op(layer_pattern)
    weight_tensor = match_result.get_tensor(weight_identity_pattern)
    if weight_tensor is None:
      weight_tensor = match_result.get_tensor(weight_resource_var_pattern)
    activation_op = match_result.get_op(layer_pattern)
    if layer_op not in matched_layer_set:
      matched_layer_set.add(layer_op)
      layer_matches.append(
          _LayerMatch(layer_op, weight_tensor, activation_op, None, None, None))

  return layer_matches

def _IsSkipLayer(activation_op):
  """Skip quantizing conv->identity->Batch norm layers.
  Args:
    activation_op: Activation op detected by layer matching pattern
  Returns:
    skip_layer: boolean, true when conv->identity->batch norm is detected.
  """

  # Exclude quantization of conv->identity->BN,
  # After folding, this part corresponds to estimation of mean and variance
  # and should not be quantized.
  skip_layer = False
  if activation_op.type == 'Identity' and len(activation_op.outputs) == 1:
    if len(activation_op.outputs[0].consumers()) == 1:
      consumer = activation_op.outputs[0].consumers()[0]
      if consumer.type == 'FusedBatchNorm':
        skip_layer = True
        logging.info(
            'Skipping quantizing %s, because it is the output of a conv/fc '
            'followed by a identity, feeding a fused batch norm.',
            activation_op.name)
  return skip_layer

class _LayerMatch(object):
  """Contains all information related to a matched Layer."""

  def __init__(self, layer_op, weight_tensor, activation_op, bypass_op,
               post_activation_bypass_op, bias_add_op):
    self._layer_op = layer_op
    self._weight_tensor = weight_tensor
    self._activation_op = activation_op
    self._bypass_op = bypass_op
    self._post_activation_bypass_op = post_activation_bypass_op
    self._bias_add_op = bias_add_op

  @property
  def layer_op(self):
    return self._layer_op

  @property
  def weight_tensor(self):
    return self._weight_tensor

  @property
  def activation_op(self):
    return self._activation_op

  @property
  def bypass_op(self):
    return self._bypass_op

  @property
  def post_activation_bypass_op(self):
    return self._post_activation_bypass_op

  @property
  def bias_add_op(self):
    return self._bias_add_op

def _InsertCalibOp(context,
                   name,
                   producer,
                   consumers,
                   vars_collection=ops.GraphKeys.GLOBAL_VARIABLES,
                   producer_scope=None,
                   consumer_scope=None):
  """Inserts calibration ops between a producer op and (multiple) consumer ops.
  Args:
    context: Context where producer and consumer operations are nested.
    name: Name for the new calibration op within the context.
    producer: Producer operation of the pairs where calibration will be
      inserted.
    consumers: Consumer operations of the pairs.
    producer_scope: The restriction of producer scope. If not None, the new op
      will be inserted only when the producer is in this scope.
    consumer_scope: The restriction of consumer scope. If not None, the new op
      will be inserted only when all the consumers are in this scope.
  Raises:
    ValueError: When producer operation is not directly connected to the
      consumer operation.
  """
  if producer_scope and not producer.name.startswith(producer_scope):
    logging.info(
        '_InsertCalibOp ignores context="%s" name="%s" '
        'because producer "%s" is not in scope "%s"',
        context, name, producer.name, producer_scope)
    return

  if consumer_scope:
    consumers_in_scope = []
    for consumer in consumers:
      if consumer.name.startswith(consumer_scope):
        consumers_in_scope.append(consumer)
      else:
        logging.info(
            '_InsertCalibOp context="%s" name="%s" ignores '
            'consumer "%s" because it is not in scope "%s"',
            context, name, consumer.name, consumer_scope)
        return
    consumers = consumers_in_scope

  name_prefix = _AddContextToName(context, name)

  name_scope = ops.get_name_scope()
  if name_scope:
    name_prefix = common.DropStringPrefix(name_prefix, name_scope + '/')

  inputs = producer.outputs[0]
  # Prevent ops from being modified multiple times. Bypass ops can sometimes
  # overlap between multiple matches, so we need to ensure that we don't
  # add duplicate calibration operations.
  #if _FollowedByFakeQuant(inputs):
  #  return

  with variable_scope.variable_scope(
      None, default_name=name_prefix, values=[inputs]) as scope:
    # Currently no per channel.
    min_max_shape = []
    vars_collections = [vars_collection] if vars_collection else []
    min_var = _ModelVariable(
        'min',
        shape=min_max_shape,
        initializer=init_ops.constant_initializer(float('inf')),
        collections=vars_collections,
        trainable=False)
    max_var = _ModelVariable(
        'max',
        shape=min_max_shape,
        initializer=init_ops.constant_initializer(-float('inf')),
        collections=vars_collections,
        trainable=False)
    batch_min = math_ops.reduce_min(inputs, name='BatchMin')
    batch_max = math_ops.reduce_max(inputs, name='BatchMax')

    range_min = math_ops.minimum(batch_min,
            min_var,
            name=name_prefix + '/range_min')
    range_max = math_ops.maximum(batch_max,
            max_var,
            name=name_prefix + '/range_max')

  return range_min, range_max

def _GetContextFromOp(op):
  """Gets the root context name from the op name."""
  context_re = re.search(r'^(.*)/([^/]+)', op.name)
  if context_re:
    return context_re.group(1)
  return ''


def _AddContextToName(context, name):
  """Adds the context to the name if it exists."""
  if not context:
    return name
  return context + '/' + name

def _ModelVariable(name,
                   shape=None,
                   initializer=None,
                   collections=None,
                   trainable=None):
  collections = list(collections or [])
  collections += [ops.GraphKeys.GLOBAL_VARIABLES]
  return variable_scope.get_variable(
      name,
      shape=shape,
      initializer=initializer,
      collections=collections,
      trainable=trainable)
