from operator import mod
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import tqdm
import torch
from hope import Hope
from utils import OCFDataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from waymo_open_dataset.protos import occupancy_flow_metrics_pb2
from google.protobuf import text_format
from waymo_open_dataset.utils import occupancy_flow_grids
from waymo_open_dataset.utils import occupancy_flow_vis
from waymo_open_dataset.utils import occupancy_flow_metrics
import tensorflow as tf
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from utils import OCFDataset,PickleOCFDataset
import numpy as np
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union
import math
from collections import OrderedDict
import torch.nn as nn
config = occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig()
config_text = """
    num_past_steps: 10
    num_future_steps: 80
    num_waypoints: 8
    cumulative_waypoints: false
    normalize_sdc_yaw: true
    grid_height_cells: 768
    grid_width_cells: 768
    sdc_y_in_grid: 192
    sdc_x_in_grid: 128
    pixels_per_meter: 6.4
    agent_points_per_side_length: 48
    agent_points_per_side_width: 16
    """
text_format.Parse(config_text, config)

NUM_PRED_CHANNELS = 4
def _get_pred_waypoint_logits(
        model_outputs) -> occupancy_flow_grids.WaypointGrids:
    """Slices model predictions into occupancy and flow grids."""
    pred_waypoint_logits = occupancy_flow_grids.WaypointGrids()

    # Slice channels into output predictions.
    for k in range(config.num_waypoints):
        index = k * NUM_PRED_CHANNELS
        waypoint_channels = model_outputs[:, :, :, index:index + NUM_PRED_CHANNELS]
        pred_observed_occupancy = waypoint_channels[:, :, :, :1]
        pred_occluded_occupancy = waypoint_channels[:, :, :, 1:2]
        pred_flow = waypoint_channels[:, :, :, 2:]
        pred_waypoint_logits.vehicles.observed_occupancy.append(
            pred_observed_occupancy)
        pred_waypoint_logits.vehicles.occluded_occupancy.append(
            pred_occluded_occupancy)
        pred_waypoint_logits.vehicles.flow.append(pred_flow)

    return pred_waypoint_logits
def _apply_sigmoid_to_occupancy_logits(
    pred_waypoint_logits: occupancy_flow_grids.WaypointGrids
) -> occupancy_flow_grids.WaypointGrids:
  """Converts occupancy logits to probabilities."""
  pred_waypoints = occupancy_flow_grids.WaypointGrids()
  pred_waypoints.vehicles.observed_occupancy = [
      tf.sigmoid(x) for x in pred_waypoint_logits.vehicles.observed_occupancy
  ]
  pred_waypoints.vehicles.occluded_occupancy = [
      tf.sigmoid(x) for x in pred_waypoint_logits.vehicles.occluded_occupancy
  ]
  pred_waypoints.vehicles.flow = pred_waypoint_logits.vehicles.flow
  return pred_waypoints
def create_animation(images, interval=100):
    plt.ioff()
    fig, ax = plt.subplots()
    dpi = 100
    size_inches = 1000 / dpi
    fig.set_size_inches([size_inches, size_inches])
    plt.ion()

    def animate_func(i):
        ax.imshow(images[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid('off')

    anim = animation.FuncAnimation(
      fig, animate_func, frames=len(images), interval=interval)
    plt.close(fig)
    return anim
def create_animation_flow(images,flows,interval=100):
  """ Creates a Matplotlib animation of the given images.

  Args:
    images: A list of numpy arrays representing the images.
    interval: Delay between frames in milliseconds.

  Returns:
    A matplotlib.animation.Animation.

  Usage:
    anim = create_animation(images)
    anim.save('/tmp/animation.avi')
    HTML(anim.to_html5_video())
  """

  plt.ioff()
  fig, ax = plt.subplots()
  dpi = 100
  size_inches = 1000 / dpi
  fig.set_size_inches([size_inches, size_inches])
  plt.ion()

  def animate_func(i):
    flow_shape = flows[i].shape
    height = flow_shape[-3]
    width = flow_shape[-2]
    flow_flat = tf.reshape(flows[i], (-1, height, width, 2))
    dx = tf.cast(flow_flat[..., 0],tf.float64)
    dy = tf.cast(flow_flat[..., 1],tf.float64)
    X, Y = np.meshgrid(np.arange(0,768,16),np.arange(0,768,16))
#     dx=magnitudes*np.cos(tf.math.atan2(dy, -dx))
#     dy=magnitudes*np.sin(tf.math.atan2(dy, -dx))
    #dx_new=magnitudes*np.cos(tf.math.atan2(dx, -dy))
    #dy_new=magnitudes*np.sin(tf.math.atan2(dx, -dy))
    dx_new = -dy
    dy_new = dx
#     a = -dy[0]
    a=dx_new[0]
    b = tf.reshape(a,[48,16,48,16])
    c = tf.transpose(b,[0,2,1,3])
    d = tf.reshape(tf.reshape(c,[-1,16,16]),[48,48,-1])
    U=tf.reduce_mean(d,axis=2)
    q=dy_new[0]
#     q = -dx[0]
    w = tf.reshape(q,[48,16,48,16])
    e = tf.transpose(w,[0,2,1,3])
    r = tf.reshape(tf.reshape(e,[-1,16,16]),[48,48,-1])
    V = tf.reduce_mean(r,axis=2)
    ax.imshow(images[i])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.quiver(X,Y,U,V, scale_units='xy', scale=1)

    ax.grid('off')

  anim = animation.FuncAnimation(
      fig, animate_func, frames=len(images), interval=interval)
  plt.close(fig)
  return anim
def flow_rgb_image_new(
    flow: tf.Tensor,
    roadgraph_image: tf.Tensor,
    agent_trails: tf.Tensor,
) -> tf.Tensor:
  """Converts (dx, dy) flow to RGB image.

  Args:
    flow: [batch_size, height, width, 2] float32 tensor holding (dx, dy) values.
    roadgraph_image: Road graph image [batch_size, height, width, 1] float32.
    agent_trails: [batch_size, height, width, 1] float32 tensor containing
      rendered trails for all agents over the past and current time frames.

  Returns:
    [batch_size, height, width, 3] float32 RGB image.
  """
  # Swap x, y for compatibilty with published visualizations.
  flow = tf.roll(flow, shift=1, axis=-1)
  
  # saturate_magnitude=-1 normalizes highest intensity to largest magnitude.
  flow_image = _optical_flow_to_rgb(flow, saturate_magnitude=-1)
  # Add roadgraph.
  flow_image = _add_grayscale_layer(roadgraph_image, flow_image)  # Black.
#   # Overlay agent trails.
  flow_image = _add_grayscale_layer(agent_trails * 0.2, flow_image)  # 0.2 alpha
  
  return flow_image,flow
def _add_grayscale_layer(
    fg_a: tf.Tensor,
    scene_rgb: tf.Tensor,
) -> tf.Tensor:
  """Adds a black/gray layer using fg_a as alpha over an RGB image."""
  # Create a black layer matching dimensions of fg_a.
  black = tf.zeros_like(fg_a)
  black = tf.concat([black, black, black], axis=-1)
  # Add the black layer with transparency over the scene_rgb image.
  overlay, _ = _alpha_blend(fg=black, bg=scene_rgb, fg_a=fg_a, bg_a=1.0)
  return overlay

def _optical_flow_to_rgb(
    flow: tf.Tensor,
    saturate_magnitude: float = -1.0,
    name: Optional[str] = None,
) -> tf.Tensor:
  """Visualize an optical flow field in RGB colorspace."""
  name = name or 'OpticalFlowToRGB'
  hsv = _optical_flow_to_hsv(flow, saturate_magnitude, name)
  return tf.image.hsv_to_rgb(hsv)
def _optical_flow_to_hsv(
    flow: tf.Tensor,
    saturate_magnitude: float = -1.0,
    name: Optional[str] = None,
) -> tf.Tensor:
  """Visualize an optical flow field in HSV colorspace.

  This uses the standard color code with hue corresponding to direction of
  motion and saturation corresponding to magnitude.

  The attr `saturate_magnitude` sets the magnitude of motion (in pixels) at
  which the color code saturates. A negative value is replaced with the maximum
  magnitude in the optical flow field.

  Args:
    flow: A `Tensor` of type `float32`. A 3-D or 4-D tensor storing (a batch of)
      optical flow field(s) as flow([batch,] i, j) = (dx, dy). The shape of the
      tensor is [height, width, 2] or [batch, height, width, 2] for the 4-D
      case.
    saturate_magnitude: An optional `float`. Defaults to `-1`.
    name: A name for the operation (optional).

  Returns:
    An tf.float32 HSV image (or image batch) of size [height, width, 3]
    (or [batch, height, width, 3]) compatible with tensorflow color conversion
    ops. The hue at each pixel corresponds to direction of motion. The
    saturation at each pixel corresponds to the magnitude of motion relative to
    the `saturate_magnitude` value. Hue, saturation, and value are in [0, 1].
  """
  with tf.name_scope(name or 'OpticalFlowToHSV'):
    flow_shape = flow.shape
    if len(flow_shape) < 3:
      raise ValueError('flow must be at least 3-dimensional, got'
                       f' `{flow_shape}`')
    if flow_shape[-1] != 2:
      raise ValueError(f'flow must have innermost dimension of 2, got'
                       f' `{flow_shape}`')
    height = flow_shape[-3]
    width = flow_shape[-2]
    flow_flat = tf.reshape(flow, (-1, height, width, 2))
    dx = flow_flat[..., 0]
    dy = flow_flat[..., 1]
    # [batch_size, height, width]
    magnitudes = tf.sqrt(tf.square(dx) + tf.square(dy))
    if saturate_magnitude < 0:
      # [batch_size, 1, 1]
      local_saturate_magnitude = tf.reduce_max(
          magnitudes, axis=(1, 2), keepdims=True)
    else:
      local_saturate_magnitude = tf.convert_to_tensor(saturate_magnitude)

    # Hue is angle scaled to [0.0, 1.0).
    hue = (tf.math.mod(tf.math.atan2(dy, dx), (2 * math.pi))) / (2 * math.pi)
    # Saturation is relative magnitude.
    relative_magnitudes = tf.math.divide_no_nan(magnitudes,
                                                local_saturate_magnitude)
    saturation = tf.minimum(
        relative_magnitudes,
        1.0  # Larger magnitudes saturate.
    )
    # Value is fixed.
    value = tf.ones_like(saturation)
    hsv_flat = tf.stack((hue, saturation, value), axis=-1)
    return tf.reshape(hsv_flat, flow_shape.as_list()[:-1] + [3])
def _alpha_blend(
    fg: tf.Tensor,
    bg: tf.Tensor,
    fg_a: Optional[tf.Tensor] = None,
    bg_a: Optional[Union[tf.Tensor, float]] = None,
) -> Tuple[tf.Tensor, tf.Tensor]:
  """Overlays foreground and background image with custom alpha values.

  Implements alpha compositing using Porter/Duff equations.
  https://en.wikipedia.org/wiki/Alpha_compositing

  Works with 1-channel or 3-channel images.

  If alpha values are not specified, they are set to the intensity of RGB
  values.

  Args:
    fg: Foreground: float32 tensor shaped [batch, grid_height, grid_width, d].
    bg: Background: float32 tensor shaped [batch, grid_height, grid_width, d].
    fg_a: Foreground alpha: float32 tensor broadcastable to fg.
    bg_a: Background alpha: float32 tensor broadcastable to bg.

  Returns:
    Output image: tf.float32 tensor shaped [batch, grid_height, grid_width, d].
    Output alpha: tf.float32 tensor shaped [batch, grid_height, grid_width, d].
  """
  if fg_a is None:
    fg_a = fg
  if bg_a is None:
    bg_a = bg
  eps = tf.keras.backend.epsilon()
  out_a = fg_a + bg_a * (1 - fg_a)
  out_rgb = (fg * fg_a + bg * bg_a * (1 - fg_a)) / (out_a + eps)
  return out_rgb, out_a
def inference():
    model=Hope()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    print(device)
    model = nn.DataParallel(model).to(device)

    model_state=torch.load("/home/yu/workspace/occupancy_flow_predict/models/HOPE13.768_09-08-2022_19-05-08.pth")
    # model.load_state_dict(model_state['net'])
    # new_state_dict = OrderedDict()
    # for k, v in model_state['net'].items():
    #   name = k[7:] # module字段在最前面，从第7个字符开始就可以去掉module
    #   new_state_dict[name] = v #新字典的key值对应的value一一对应
    # model.load_state_dict(new_state_dict)
    model.load_state_dict(model_state['net'])

    # model.eval().cuda()
    model.eval()

    test_dataset =PickleOCFDataset(file_path="/data/hdd/yjn/waymo_data/test_ocf_acc")
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=16)
    name=1   
    if not os.path.exists('/home/yu/workspace/anim'):
        os.mkdir('/home/yu/workspace/anim')
    with torch.no_grad():
        for data in tqdm(testloader):

            vehicles_current_occupancy = data['vehicles_current_occupancy']
            vehicles_past_occupancy = data['vehicles_past_occupancy']

            pedestrians_current_occupancy = data['pedestrians_current_occupancy']

            pedestrians_past_occupancy = data['pedestrians_past_occupancy']

            cyclists_current_occupancy = data['cyclists_current_occupancy']

            cyclists_past_occupancy = data['cyclists_past_occupancy']

            vis_grids_roadgraph = data['vis_grids_roadgraph']
            vis_grids_agent_trails = data['vis_grids_agent_trails']

       
            model_inputs = torch.cat(
            (
             vehicles_past_occupancy,
             vehicles_current_occupancy,
             torch.clamp((pedestrians_past_occupancy + cyclists_past_occupancy), 0, 1),
             torch.clamp((pedestrians_current_occupancy + cyclists_current_occupancy), 0, 1),
             vis_grids_roadgraph,
             ),
            dim=-1,
            )

            model_inputs = model_inputs.permute(0, 3, 1, 2).contiguous().to(device)
            model_outputs=model(model_inputs)
            model_outputs=model_outputs.cpu().detach().numpy()
            model_outputs=tf.convert_to_tensor(model_outputs)
            pred_waypoint_logits = _get_pred_waypoint_logits(model_outputs)
            pred_waypoints = _apply_sigmoid_to_occupancy_logits(pred_waypoint_logits)
            
            ### visiual 
            images = []
            for k in range(config.num_waypoints):
                observed_occupancy_grids = pred_waypoints.get_observed_occupancy_at_waypoint(
                k)
                observed_occupancy_rgb = occupancy_flow_vis.occupancy_rgb_image(
                agent_grids=observed_occupancy_grids,
                roadgraph_image=vis_grids_roadgraph,
                gamma=1.6,
                 )
                images.append(observed_occupancy_rgb[0])
        
            if not os.path.exists('/home/yu/workspace/anim/observed_occupancy_rgb'):
                os.mkdir('/home/yu/workspace/anim/observed_occupancy_rgb')
            anim = create_animation(images, interval=200)
            anim.save("/home/yu/workspace/anim/observed_occupancy_rgb/Observed occupancy_{}.gif".format(name), writer='pillow')
            images = []
            for k in range(config.num_waypoints):
                occluded_occupancy_grids = pred_waypoints.get_occluded_occupancy_at_waypoint(
                k)
                occluded_occupancy_rgb = occupancy_flow_vis.occupancy_rgb_image(
                agent_grids=occluded_occupancy_grids,
                roadgraph_image=vis_grids_roadgraph,
                gamma=1.6,
                )
                images.append(observed_occupancy_rgb[0])
            if not os.path.exists('/home/yu/workspace/anim/occluded_occupancy_rgb/'):
                os.mkdir('/home/yu/workspace/anim/occluded_occupancy_rgb/')
            anim2 = create_animation(images, interval=200)
        
            anim2.save("/home/yu/workspace/anim/occluded_occupancy_rgb/occluded_occupancy_{}.gif".format(name), writer='pillow')
            images = []
            for k in range(config.num_waypoints):
                flow_rgb = occupancy_flow_vis.flow_rgb_image(
                flow=pred_waypoints.vehicles.flow[k],
                roadgraph_image=vis_grids_roadgraph,
                agent_trails=vis_grids_agent_trails,
                )
                images.append(flow_rgb[0])
            if not os.path.exists('/home/yu/workspace/anim/flow_rgb/'):
                os.mkdir('/home/yu/workspace/anim/flow_rgb/')
            anim3 = create_animation(images, interval=200)
        
            anim3.save("/home/yu/workspace/anim/flow_rgb/flow_{}.gif".format(name), writer='pillow')
            images = []
            flows=[]
            for k in range(config.num_waypoints):
                observed_occupancy_grids = pred_waypoints.get_observed_occupancy_at_waypoint(k)
                occupancy = observed_occupancy_grids.vehicles
                flow = pred_waypoints.vehicles.flow[k]
                occupancy_flow = occupancy * flow
                flow_rgb,flow= flow_rgb_image_new(
                flow=occupancy_flow,
                roadgraph_image=vis_grids_roadgraph,
                agent_trails=vis_grids_agent_trails,
                )
                images.append(flow_rgb[0])
                flows.append(flow[0])

        
            if not os.path.exists('/home/yu/workspace/anim/occupancy_flow/'):
                os.mkdir('/home/yu/workspace/anim/occupancy_flow/')
            anim4 = create_animation_flow(images,flows,interval=200)
            anim4.save("/home/yu/workspace/anim/occupancy_flow/occupancy_flow_{}.gif".format(name), writer='pillow')
            name+=1
            ######
        
            # metrics = occupancy_flow_metrics.compute_occupancy_flow_metrics(
            # config=config,
            # true_waypoints=true_waypoints,
            # pred_waypoints=pred_waypoints,
            # )

if __name__=="__main__":

    inference()
