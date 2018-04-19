import os
import sys
import glob
import random
import math
import datetime
import itertools
import json
import re
import logging
from collections import OrderedDict
import numpy as np
import scipy.misc
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.initializers as KI
import keras.engine as KE
import keras.models as KM

import utils

# Requires TensorFlow 1.3+ and Keras 2.0.8+.
from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')

############################################################
#  ROIAlign Layer
############################################################


def log2_graph(x):
    return tf.log(x)/tf.log(2.0)


class PyramidROIAlign(KE.layer):
    def __init__(self, pool_shape, image_shape, **kwargs):
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)
        self.image_shape = tuple(image_shape)
        
    def call(self, inputs):
        boxes = inputs[0]
        
        feature_maps = inputs[1:]
        #each:[batch, num_boxes, 1]
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
        h = y2 - y1
        w = x2 - x1
        
        image_area = tf.cast(
            self.image_shape[0]*self.image_shape[1], tf.float32)
        roi_level = log2_graph(tf.sqrt(h*w)/(224.0/tf.sqrt(image_area)))
        roi_level = tf.minimum(5, tf.maximum(
            2, 4+tf.cast(tf.round(roi_level), tf.int32)))
        # roi_level: [batch, num_boxes, 1], all batch hold the same roi_level?
        roi_level = tf.squeeze(roi_level, 2) # only squeeze axis=2
        
        pooled = []
        box_to_level = []
        for i, level in enumerate(range(2,6)):
            ix = tf.where(tf.equal(roi_level, level))
            level_boxes = tf.gather_nd(boxes, ix) # batch, num_level, 4
            
            box_indices = tf.cast(ix[:,0], tf.int32)
            box_to_level.append(ix)
            
            # note !!!
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)
            
            pooled.append(tf.image.crop_and_resize(
                feature_maps[i], level_boxes, box_indices, self.pool_shape,
                method="bilinear")) # crop level_boxes, then resize
            
        pooled = tf.concat(pooled, axis=0)
        
        box_to_level = tf.concat(box_to_level, axis=0)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range], 
                                 axis=1)
        
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
            box_to_level)[0]).indices[::-1]
        ix = tf.gather(box_to_level[:, 2], ix)
        pooled = tf.gather(pooled, ix)
        
        pooled = tf.expand_dims(pooled, 0)
        
        return pooled
    
    def compute_output_shape(self, input_shape):
        # return a tuple
        return input_shape[0][:2] + self.pool_shape + (input_shape[1][-1],)
    
############################################################
#  Detection Target Layer
############################################################


def overlap_graph(boxes1, boxes2):
    # boxes1, boxes2: N, 4(y1,x1,y2,x2)
    # b1: N,1,4*N2 ==> N2*N, 4, eg., [1,1,2,2,3,3]
    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),
                           [1, 1, tf.shape(boxes2)[0]]), [-1,4])
    # b2: N2*N, 4, eg., [1,2,1,2,1,2]
    b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
    
    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    intersection = tf.maximum(y2-y1, 0) * tf.maximum(x2-x1, 0)
    
    b1_area = (b1_y2-b1_y1) * (b1_x2-b1_x1)
    b2_area = (b2_y2-b2_y1) * (b2_x2-b2_x1)
    union = b1_area + b2_area - intersection
    
    iou = intersection / union
    overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]]) # N1, N2
    
    return overlaps


def detection_targets_graph(proposals, gt_class_ids, gt_boxes, gt_masks, config):
    """
    For a single image

    proposals: N, 4 normalized (x1, y1, x2, y2)
    gt_class_ids: max_instance
    gt_boxes: max_instance, 4 normalized
    gt_masks: h, w, max_instance
    """
    asserts = [tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals],
                         name="roi_assertion"),]
    
    # control_inputs: A list of Operation or Tensor objects which must be
    # executed or computed before running the operations defined in the context.
    # Can also be None to clear the control dependencies.
    with tf.control_dependencies(asserts):
        proposals = tf.identity(proposals)
        
    # remove zero padding
    proposals, _ = trim_zeros_graph(proposals, name="trim_proposals")
    gt_boxes, non_zeros = trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
    gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros,
                                   name="trim_gt_class_ids")
    gt_masks = tf.gather(gt_masks, tf.where(non_zeros)[:, 0], axis=2,
                         name="trim_gt_masks")
    
    # exclude crowd boxes
    crowd_ix = tf.where(gt_class_ids < 0)[:, 0]
    non_crowd_ix = tf.where(gt_class_ids > 0)
    crowd_boxes = tf.gather(gt_boxes, crowd_ix)
    # tf.gather indices slices along specified axis
    crowd_masks = tf.gather(gt_masks, crowd_ix, axis=2)
    gt_class_ids = tf.gather(gt_class_ids, non_crowd_ix)
    gt_boxes = tf.gather(gt_boxes, non_crowd_ix)
    gt_masks = tf.gather(gt_masks, non_crowd_ix, axis=2)
    
    # compute overlap matrix [proposal, gt_boxes]
    overlaps = overlaps_graph(proposals, gt_boxes)
    
    # overlap with crowd boxes [anchors???, crowds]
    crowd_overlaps = overlap_graph(proposals, crowd_boxes)
    crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=1)
    non_crowd_bool = (crowd_iou_max < 0.001) # ???
    
    # Determine positive and negative ROIs
    roi_iou_max = tf.reduce_max(overlaps, axis=1)
    # 1. pos rois has over 50% overlap with gt boxes
    positive_roi_bool = (roi_iou_max >= 0.5)
    positive_indices = tf.where(positive_roi_bool)[:,0]
    # 2. neg roi has less 50% overlap with gt_boxes. Skip Crowd??
    negative_indices = tf.where(tf.logical_and(roi_iou_max<0.5,
                                               non_crowd_bool))[:, 0]
    
    # Subsample ROIs. Aim for 33% positive
    # Pos ROI
    positive_count = int(config.TRIAN_ROIS_PER_IMAGE *
                         config.ROI_POSITIVE_RATIO)
    positive_indices = tf.random_shuffle(positive_indices)[:positive_count]
    positive_count = tf.shape(positive_indices)[0]
    # Neg ROI
    r = 1.0 / config.ROI_POSITIVE_RATIO
    negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), 
                             tf.int32) - positive_count
    # gather selected ROIs
    positive_rois = tf.gather(proposals, positive_indices)
    negative_rois = tf.gather(proposals, negative_indices)

    positive_overlaps = tf.gather(overlaps, positive_indices)
    roi_gt_box_assignment = tf.argmax(positive_overlaps, axis=1)
    roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
    roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)

    # box refinement for positive rois: delta ???
    deltas = utils.box_refinement_graph(positive_rois, roi_gt_boxes)
    deltas /= config.BBOX_STD_DEV

    transposed_masks = tf.expand_dims(tf.transpose(gt_masks, [2,0,1]), -1) # N, imgh, imgw, 1
    roi_masks = tf.gather(transposed_masks, roi_gt_box_assignment)

    # Compute mask targets
    boxes = positive_rois
    if config.USE_MINI_MASK: # ???
        y1, x1, y2, x2 = tf.split(positive_rois, 4, axis=1) # equally split
        gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(roi_gt_boxes, 4, axis=1)
        gt_h = gt_y2 - gt_y1
        gt_w = gt_x2 - gt_x1
        # normalize proposal ROIs coordinates based gt box
        y1 = (y1 - gt_y1) / gt_h
        x1 = (x1 - gt_x1) / gt_w
        y2 = (y2 - gt_y2) / gt_h
        x2 = (x2 - gt_x2) / gt_w
        boxes = tf.concat([x1, y1, x2, y2], 1) # N_roi, 4
    box_ids = tf.range(0, tf.shape(roi_masks)[0])
    masks = tf.image.crop_and_resize(tf.cast(roi_masks, tf.float32), boxes,
                                     box_ids, config.MINI_MASK_SHAPE)
    masks = tf.squeeze(masks, axis=3) # only keep n, h, w

    masks = tf.round(masks)

    rois = tf.concat([positive_rois, negative_rois], axis=0) # npos+nneg, 4
    N = tf.shape(negative_rois)[0]
    P = tf.maximum(config.TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0], 0) # padding size
    rois = tf.pad(rois, [(0, P), (0, 0)]) # MAX_TRAIN_ROI_PER_IMAGE, 4
    roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, N+P), (0, 0)])
    roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N+P)])
    deltas = tf.pad(deltas, [(0, N+P), (0, 0)])
    masks = tf.pad(masks, [(0, N+P), (0, 0)]) # padding-zero cropped gt masks

    return rois, roi_gt_class_ids, deltas, masks


class DetectionTargetLayer(KE.Layer):

    def __init__(self, config, **kwargs):
        super(DetectionTargetLayer, self).__init__(**kwargs)
        self.config =config

    def call(self, inputs):
        proposals, gt_class_ids, gt_boxes, gt_masks = inputs

        names = ["rois", "target_class_ids", "target_bbox", "target_mask"]
        outputs = utils.batch_slice([proposals, gt_class_ids, gt_boxes, gt_masks],
                                    lambda w, x, y, z: detection_targets_graph(
                                        w, x, y, z, self.config),
                                    self.config.IMAGES_PER_GPU, names=names)

        return outputs

    def compute_output_shape(self, input_shape):
        return [
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),
            (None, 1),
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),
            (None, self.config.TRAIN_ROIS_PER_IMAGE, self.config.MASK_SHAPE[0],
             self.config.MASK_SHAPE[1])
        ]

    def compute_mask(self, inputs, mask=None):
        return [None, None, None, None]

############################################################
#  Detection Layer
############################################################


def clip_to_window(window, boxes):
    """
    window: (y1, x1, y2, x2). The window in the image we want to clip to.
    boxes: [N, (y1, x1, y2, x2)]
    """
    boxes[:, 0] = np.maximum(np.minimum(boxes[:, 0], window[2]), window[0])
    boxes[:, 1] = np.maximum(np.minimum(boxes[:, 1], window[3]), window[1])
    boxes[:, 2] = np.maximum(np.minimum(boxes[:, 2], window[2]), window[0])
    boxes[:, 3] = np.maximum(np.minimum(boxes[:, 3], window[3]), window[1])
    return boxes

def refine_detection_graph(rois, probs, deltas, window, config):
    # Class ID per ROI
    class_ids = tf.argmax(probs, axis=1)
    # Class probebility of top class of each ROI
    indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1)
    ## gather will collect slices while gather_nd collect points/slices
    class_scores = tf.gather_nd(probs, indices)
    # Class-specific bounding box deltas [N, num_classes, (dy, dx, log(dh), log(dw))]
    deltas_specific = tf.gather_nd(deltas, indices)
    refined_rois = apply_box_deltas_graph(rois, deltas_specific*config.BBOX_STD_DEV)
    refined_rois = clip_to_window(window, refined_rois)
    refined_rois = tf.to_int32(tf.rint(refined_rois))# round and cast to int

    # Filter out bg(0) boxes
    keep = tf.where(class_ids > 0)[:, 0]
    # Filter out low confidence boxes
    if config.DETECTION_MIN_CONFIDENCE:
        conf_keep = tf.where(class_scores >= config.DETECTION_MIN_CONFIDENCE)[:, 0]
        keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                        tf.expand_dims(conf_keep, 0))
        keep = tf.sparse_tensor_to_dense(keep)[0]

    # Apply per-class NMS
    pre_nms_class_ids = tf.gather(class_ids, keep)
    pre_nms_scores = tf.gather(class_scores, keep)
    pre_nms_rois = tf.gather(refined_rois, keep)
    # tf.unique: return a tuple of (unique values, indices)
    unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

    def nms_keep_map(class_id):
        # apply nms for a given class
        ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))
        # apply nms, return 1D array with indices
        class_keep = tf.image.non_max_suppression(
            tf.to_float(tf.gather(pre_nms_rois, ixs)),
            tf.gather(pre_nms_scores, ixs),
            max_output_size=config.DETECTION_MAX_INSTANCES,
            iou_threshold=config.DETECTION_NMS_THRESHOLD
        )
        class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
        gap = config.DETECTION_MAX_INSTANCES - tf.shape(class_keep)[0]
        # why pad -1 ???
        class_keep = tf.pad(class_keep, [(0, gap)], mode='CONSTANT', constant_values=-1)
        # Set shape so map_fn() can infer result shape ???
        class_keep.set_shape([config.DETECTION_MAX_INSTANCES])
        return class_keep

    nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids, dtype=tf.int64)
    nms_keep = tf.reshape(nms_keep, [-1]) # return 1D array
    nms_keep = tf.gather(nms_keep, tf.where(nms_keep>-1)[:,0])
    # 4. Compute intersection between keep and nms_keep. Why need this step ???
    keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                    tf.expand_dims(nms_keep, 0))
    keep = tf.sparse_tensor_to_dense(keep)[0]
    # Keep top detections
    roi_count = config.DETECTION_MAX_INSTANCES
    class_scores_keep = tf.gather(class_scores, keep)
    num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
    # tf.nn.top_k: return a tuple of (values, indices)
    top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
    keep = tf.gather(keep, top_ids)

    # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
    detections = tf.concat([
        tf.to_float(tf.gather(refined_rois, keep)),
        tf.to_float(tf.gather(class_ids, keep))[..., tf.newaxis],
        tf.to_float(tf.gather(class_scores, keep))[..., tf.newaxis]
    ], axis=1)

    # Pad with zeros if detections < DETECTION_MAX_INSTANCES
    gap = config.DETECTION_MAX_INSTANCES - tf.shape(detections)[0]
    detections = tf.pad(detections, [(0, gap), (0, 0)], "CONSTANT")
    return detections


class DetectionLayer(KE.Layer):
    def __init__(self, config=None, **kwargs):
        super(DetectionLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        rois = inputs[0]
        mrcnn_class = inputs[1]
        mrcnn_bbox = inputs[2]
        image_meta = inputs[3]

        _, _, window, _ = parse_image_meta_graph(image_meta)
        detections_batch = utils.batch_slice(
            [rois, mrcnn_class, mrcnn_bbox, window],
            lambda x, y, w, z: refine_detection_graph(x, y, w, z, self.config),
            self.config.IMAGES_PER_GPU
        ) # return (batch_size*max_instances, 6) ???

        # reshape output
        return tf.reshape(
            detections_batch,
            [self.config.BATCH_SIZE, self.config.DETECTION_MAX_INSTANCES, 6]
        )

    def compute_output_shape(self, input_shape):
        return (None, self.config.DETECTION_MAX_INSTANCES, 6)


# Region Proposal Network (RPN)
def rpn_graph(feature_map, anchors_per_location, anchor_stride):
    """
    Builds the computation graph of region proposal graph

    :param feature_map: backbone output [batch, h, w, channels]
    :param anchors_per_location: number of anchors per pixel in feature_map
    :param anchor_stride: the density of anchor. It is typically 1 per pixel
    or 1 every 2 pixel.
    :return:
    rpn_logits: [batch, h, w, 2] anchor classifier logits before softmax
    rpn_probs: [batch, h, w, 2] anchor classifier probs
    rpn_bbox: [batch, h, w, (dy, dx, log(dh), log(dw))] Deltas

    # Shared convolutional base of RPN
    """
    shared = KL.Conv2D(512, (3, 3), padding='same', activation='relu',
                       strides=anchor_stride,
                       name='rpn_conv_shared')(feature_map)

    # Anchor Score. [batch, h, w, anchors per location*2]
    x = KL.Conv2D(2*anchors_per_location, (1, 1), padding='valid',
                  activation='linear', name='rpn_class_raw')(shared)

    # Reshape [batch, anchors, 2].
    # Why KL.Lambda here rather than tf.reshape?
    # Due to difference between keras and tensorflow tensor
    rpn_class_logits = KL.Lambda(
        lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2])
    )(x)

    # Softmax on last dimension of BG/FG
    rpn_probs = KL.Activation(
        'softmax', name='rpn_class_xxx'
    )(rpn_class_logits)

    x = KL.Conv2D(anchors_per_location*4, (1, 1), padding='valid',
                  activation='linear', name='rpn_bbox_pred')(shared)
    # reshpe to [batch, anchors, 4]
    rpn_bbox = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(x)

    return  [rpn_class_logits, rpn_probs, rpn_bbox]

def build_rpn_model(anchor_stride, anchors_per_location, depth):
    """
    Build Keras model for RPN.
    It wraps the RPN graph so it can be used multiple times with
    shared weights.

    :param anchor_stride:
    :param anchors_per_location:
    :param depth:
    :return:
    a keras model object
    """
    input_feature_map = KL.Input(shape=[None, None, depth],
                                 name='input_rpn_feature_map')
    outputs = rpn_graph(input_feature_map, anchors_per_location, anchor_stride)
    return KM.Model([input_feature_map], outputs, name='rpn_model')


############################################################
#  Feature Pyramid Network Heads
############################################################
def fpn_classifier_graph(rois, feature_maps,
                         image_shape, pool_size, num_classes):
    """

    :param rois:
    :param feature_maps: List of feature maps from diffent layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    :param image_shape:
    :param pool_size:
    :param num_classes:
    :return:
    """
    # [batch, num_boxes, pool_h, pool_w, channels]
    x = PyramidROIAlign([pool_size, pool_size], image_shape,
                        name='roi_align_classifier')([rois] + feature_maps)
    # Two 1024 FC layers (implemented with Conv2D for consistency)
    # shape: [batch, num_boxes, 1, 1, channels]
    x = KL.TimeDistributed(KL.Conv2D(1024, (pool_size, pool_size), padding='valid'),
                           name='mrcnn_class_conv1')(x)
    x = KL.TimeDistributed(BatchNorm(axis=3), name='mrcnn_class_bn1')(x)
    x = KL.Activation('relu')(x)
    x = KL.TimeDistributed(KL.Conv2D(1024, (1, 1)),
                           name="mrcnn_class_conv2")(x)
    x = KL.TimeDistributed(BatchNorm(axis=3),
                           name='mrcnn_class_bn2')(x)
    x = KL.Activation('relu')(x)
    # why separated K.squeeze here?
    # shared: [batch, num_boxes, 1024]
    shared = KL.Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2),
                       name='pool_squeeze')(x)

    # Classifier head
    # shape: [batch, num_boxes, num_classes]
    mrcnn_class_logits = KL.TimeDistributed(KL.Dense(num_classes),
                                            name='mrcnn_class_logits')(shared)
    mrcnn_probs = KL.TimeDistributed(KL.Activation('softmax'),
                                     name='mrcnn_class')(mrcnn_class_logits)

    # BBox head
    # [batch, boxes, num_classes * (dy, dx, log(dh), log(dw))]
    x = KL.TimeDistributed(KL.Dense(num_classes*4, activation='linear'),
                           name='mrcnn_bbox_fc')(shared)
    s = K.int_shape(x)
    mrcnn_bbox = KL.Reshape((s[1], num_classes, 4), name='mrcnn_bbox')(x)

    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox


def build_fpn_mask_graph(rois, feature_maps,
                         image_shape, pool_size, num_classes):
    """
    Mask/Segmentation head comprises 4 CNN(conv+bn+relu) layers.

    :param rois:
    :param feature_maps: a list of [P2, P3, P4, P5]
    :param image_shape:
    :param pool_size:
    :param num_classes:
    :return:
    """
    x = PyramidROIAlign([pool_size, pool_size], image_shape,
                        name='roi_align_classifier')([rois]+feature_maps)
    # conv1
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding='same'),
                           name='mrcnn_mask_conv1')(x)
    x = KL.TimeDistributed(BatchNorm(axis=3),
                           name='mrcnn_mask_bn1')(x)
    x = KL.Activation('relu')(x)
    # conv2
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv2")(x)
    x = KL.TimeDistributed(BatchNorm(axis=3),
                           name='mrcnn_mask_bn2')(x)
    x = KL.Activation('relu')(x)
    # conv3
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv3")(x)
    x = KL.TimeDistributed(BatchNorm(axis=3),
                           name='mrcnn_mask_bn3')(x)
    x = KL.Activation('relu')(x)
    # conv4
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv4")(x)
    x = KL.TimeDistributed(BatchNorm(axis=3),
                           name='mrcnn_mask_bn4')(x)
    x = KL.Activation('relu')(x)

    # output
    x = KL.TimeDistributed(KL.Conv2DTranspose(256, (2, 2), strides=2, activation='relu'),
                           name='mrcnn_mask_deconv')(x)
    x = KL.TimeDistributed(KL.Conv2D(num_classes, (1, 1), strides=1, activation='sigmoid'),
                           name='mrcnn_mask')(x)

    return x


############################################################
#  Loss Functions
############################################################
def smooth_l1_loss(y_true, y_pred):
    """

    :param y_true: [N, 4], could by any shape
    :param y_pred: the same as y_true
    :return:
    """
    diff = K.abs(y_pred- y_true)
    less_than_one = K.cast(K.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
    return  loss


def rpn_class_loss_graph(rpn_match, rpn_class_logits):
    """RPN anchor classifier loss.

    :param rpn_match: [batch, anchors, 1]
    :param rpn_class_logits: [batch, anchors, 2]
    :return:
    """
    # squeeze last dimension
    rpn_match = tf.squeeze(rpn_match, -1)
    # get anchor classes: +1/-1 ==> 1/0
    anchor_class = K.cast(K.equal(rpn_match, 1), tf.int32)
    indices = tf.where(K.not_equal(rpn_match, 0)) # positive

    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices) # prediction
    anchor_class = tf.gather_nd(anchor_class, indices) # ground truth
    # from_logits: Boolean, whether output
    # is the result of a softmax, or is a tensor of logits.
    loss = K.sparse_categorical_crossentropyfr(target=anchor_class,
                                             output=rpn_class_logits,
                                             from_logits=True)
    # return 0 if loss is none
    loss = K.switch(tf.size(loss)>0, K.mean(loss), tf.constant(0.0))
    return loss


def rpn_bbox_loss_graph(config, target_bbox, rpn_match, rpn_bbox):
    """

    :param config:
    :param target_bbox: target_bbox: [batch, max positive anchors,
    (dy, dx, log(dh), log(dw))].
    :param rpn_match:
    :param rpn_bbox:
    :return:
    """
    # Positive anchors contribute to loss but the negative and neutral anchors don't
    rpn_match = K.squeeze(rpn_match, -1)
    indices = tf.where(K.equal(rpn_match, 1))

    rpn_bbox = tf.gather_nd(rpn_bbox, indices)
    # Trim target bounding box deltas to the same length as rpn_bbox.
    batch_counts = K.sum(K.cast(K.equal(rpn_match, 1), tf.int32), axis=1)
    target_bbox = batch_pack_graph(target_bbox, batch_counts, config.IMAGES_PER_GPU)

    loss = smooth_l1_loss(target_bbox, rpn_bbox)

    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    return loss


def mrcnn_class_loss_graph(target_class_ids, pred_class_logits,
                           active_class_ids):
    target_class_ids = tf.cast(target_class_ids, 'int64')
    pred_class_ids =










































    
    
    
    
    
    
    
    
        
    
    
    
    
    
    
    
    
        
        
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        