"""
Source code of the Fully Convolutional Feature Descriptor (ICCV 2019), based on the FCGF repository: https://github.com/chrischoy/FCGF/

If you use in your project please consider also citing: https://node1.chrischoy.org/data/publications/fcgf/fcgf.pdf

"""

# -*- coding: future_fstrings -*-
import torch
import torch.nn as nn
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF

def get_norm(norm_type, num_feats, bn_momentum=0.05, D=-1):
  if norm_type == 'BN':
    return ME.MinkowskiBatchNorm(num_feats, momentum=bn_momentum)
  elif norm_type == 'IN':
    return ME.MinkowskiInstanceNorm(num_feats, dimension=D)
  else:
    raise ValueError(f'Type {norm_type}, not defined')


class BasicBlockBase(nn.Module):
  expansion = 1
  NORM_TYPE = 'BN'

  def __init__(self,
               inplanes,
               planes,
               stride=1,
               dilation=1,
               downsample=None,
               bn_momentum=0.1,
               D=3):
    super(BasicBlockBase, self).__init__()

    self.conv1 = ME.MinkowskiConvolution(
        inplanes, planes, kernel_size=3, stride=stride, dimension=D)
    self.norm1 = get_norm(self.NORM_TYPE, planes, bn_momentum=bn_momentum, D=D)
    self.conv2 = ME.MinkowskiConvolution(
        planes,
        planes,
        kernel_size=3,
        stride=1,
        dilation=dilation,
        has_bias=False,
        dimension=D)
    self.norm2 = get_norm(self.NORM_TYPE, planes, bn_momentum=bn_momentum, D=D)
    self.downsample = downsample

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.norm1(out)
    out = MEF.relu(out)

    out = self.conv2(out)
    out = self.norm2(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = MEF.relu(out)

    return out


class BasicBlockBN(BasicBlockBase):
  NORM_TYPE = 'BN'


class BasicBlockIN(BasicBlockBase):
  NORM_TYPE = 'IN'


def get_block(norm_type,
              inplanes,
              planes,
              stride=1,
              dilation=1,
              downsample=None,
              bn_momentum=0.1,
              D=3):
  if norm_type == 'BN':
    return BasicBlockBN(inplanes, planes, stride, dilation, downsample, bn_momentum, D)
  elif norm_type == 'IN':
    return BasicBlockIN(inplanes, planes, stride, dilation, downsample, bn_momentum, D)
  else:
    raise ValueError(f'Type {norm_type}, not defined')





class FCGFNet(ME.MinkowskiNetwork):
  NORM_TYPE = 'BN'
  BLOCK_NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256]
  TR_CHANNELS = [None, 64, 64, 64, 128]

    # To use the model, must call initialize_coords before forward pass.
  # Once data is processed, call clear to reset the model before calling initialize_coords
  def __init__(self,
               in_channels=1,
               out_channels=32,
               bn_momentum=0.05,
               normalize_feature=True,
               conv1_kernel_size=7,
               D=3):
    ME.MinkowskiNetwork.__init__(self, D)
    NORM_TYPE = self.NORM_TYPE
    BLOCK_NORM_TYPE = self.BLOCK_NORM_TYPE
    CHANNELS = self.CHANNELS
    TR_CHANNELS = self.TR_CHANNELS
    self.normalize_feature = normalize_feature
    self.conv1 = ME.MinkowskiConvolution(
        in_channels=in_channels,
        out_channels=CHANNELS[1],
        kernel_size=conv1_kernel_size,
        stride=1,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.norm1 = get_norm(NORM_TYPE, CHANNELS[1], bn_momentum=bn_momentum, D=D)

    self.block1 = get_block(
        BLOCK_NORM_TYPE, CHANNELS[1], CHANNELS[1], bn_momentum=bn_momentum, D=D)

    self.conv2 = ME.MinkowskiConvolution(
        in_channels=CHANNELS[1],
        out_channels=CHANNELS[2],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.norm2 = get_norm(NORM_TYPE, CHANNELS[2], bn_momentum=bn_momentum, D=D)

    self.block2 = get_block(
        BLOCK_NORM_TYPE, CHANNELS[2], CHANNELS[2], bn_momentum=bn_momentum, D=D)

    self.conv3 = ME.MinkowskiConvolution(
        in_channels=CHANNELS[2],
        out_channels=CHANNELS[3],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.norm3 = get_norm(NORM_TYPE, CHANNELS[3], bn_momentum=bn_momentum, D=D)

    self.block3 = get_block(
        BLOCK_NORM_TYPE, CHANNELS[3], CHANNELS[3], bn_momentum=bn_momentum, D=D)

    self.conv4 = ME.MinkowskiConvolution(
        in_channels=CHANNELS[3],
        out_channels=CHANNELS[4],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.norm4 = get_norm(NORM_TYPE, CHANNELS[4], bn_momentum=bn_momentum, D=D)

    self.block4 = get_block(
        BLOCK_NORM_TYPE, CHANNELS[4], CHANNELS[4], bn_momentum=bn_momentum, D=D)

    self.conv4_tr = ME.MinkowskiConvolutionTranspose(
        in_channels=CHANNELS[4],
        out_channels=TR_CHANNELS[4],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.norm4_tr = get_norm(NORM_TYPE, TR_CHANNELS[4], bn_momentum=bn_momentum, D=D)

    self.block4_tr = get_block(
        BLOCK_NORM_TYPE, TR_CHANNELS[4], TR_CHANNELS[4], bn_momentum=bn_momentum, D=D)

    self.conv3_tr = ME.MinkowskiConvolutionTranspose(
        in_channels=CHANNELS[3] + TR_CHANNELS[4],
        out_channels=TR_CHANNELS[3],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.norm3_tr = get_norm(NORM_TYPE, TR_CHANNELS[3], bn_momentum=bn_momentum, D=D)

    self.block3_tr = get_block(
        BLOCK_NORM_TYPE, TR_CHANNELS[3], TR_CHANNELS[3], bn_momentum=bn_momentum, D=D)

    self.conv2_tr = ME.MinkowskiConvolutionTranspose(
        in_channels=CHANNELS[2] + TR_CHANNELS[3],
        out_channels=TR_CHANNELS[2],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.norm2_tr = get_norm(NORM_TYPE, TR_CHANNELS[2], bn_momentum=bn_momentum, D=D)

    self.block2_tr = get_block(
        BLOCK_NORM_TYPE, TR_CHANNELS[2], TR_CHANNELS[2], bn_momentum=bn_momentum, D=D)

    self.conv1_tr = ME.MinkowskiConvolution(
        in_channels=CHANNELS[1] + TR_CHANNELS[2],
        out_channels=TR_CHANNELS[1],
        kernel_size=1,
        stride=1,
        dilation=1,
        has_bias=False,
        dimension=D)

    # self.block1_tr = BasicBlockBN(TR_CHANNELS[1], TR_CHANNELS[1], bn_momentum=bn_momentum, D=D)

    self.final = ME.MinkowskiConvolution(
        in_channels=TR_CHANNELS[1],
        out_channels=out_channels,
        kernel_size=1,
        stride=1,
        dilation=1,
        has_bias=True,
        dimension=D)

  def forward(self, x):
    out_s1 = self.conv1(x)
    out_s1 = self.norm1(out_s1)
    out_s1 = self.block1(out_s1)
    out = MEF.relu(out_s1)

    out_s2 = self.conv2(out)
    out_s2 = self.norm2(out_s2)
    out_s2 = self.block2(out_s2)
    out = MEF.relu(out_s2)

    out_s4 = self.conv3(out)
    out_s4 = self.norm3(out_s4)
    out_s4 = self.block3(out_s4)
    out = MEF.relu(out_s4)

    out_s8 = self.conv4(out)
    out_s8 = self.norm4(out_s8)
    out_s8 = self.block4(out_s8)
    out = MEF.relu(out_s8)

    out = self.conv4_tr(out)
    out = self.norm4_tr(out)
    out = self.block4_tr(out)
    out_s4_tr = MEF.relu(out)

    out = ME.cat(out_s4_tr, out_s4)

    out = self.conv3_tr(out)
    out = self.norm3_tr(out)
    out = self.block3_tr(out)
    out_s2_tr = MEF.relu(out)

    out = ME.cat(out_s2_tr, out_s2)

    out = self.conv2_tr(out)
    out = self.norm2_tr(out)
    out = self.block2_tr(out)
    out_s1_tr = MEF.relu(out)

    out = ME.cat(out_s1_tr, out_s1)
    out = self.conv1_tr(out)
    out = MEF.relu(out)
    out = self.final(out)

    if self.normalize_feature:
      return ME.SparseTensor(
          out.F / torch.norm(out.F, p=2, dim=1, keepdim=True),
          coords_key=out.coords_key,
          coords_manager=out.coords_man)
    else:
      return out

