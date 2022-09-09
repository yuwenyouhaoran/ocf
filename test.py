from operator import mod
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
from tools.create_ground_truth_timestep_gris_new import  create_ground_truth_timestep_grids
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


def inference():
    model=Hope()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    print(device)
    model = nn.DataParallel(model).to(device)

    model_state=torch.load("/home/yu/workspace/occupancy_flow_predict/models/HOPE19.345_09-07-2022_21-38-13.pth")
    # print('model_state',model_state['net'].keys())
    # model.load_state_dict(model_state['net'])
    # new_state_dict = OrderedDict()
    # for k, v in model_state['net'].items():
    #   name = k[7:] # module字段在最前面，从第7个字符开始就可以去掉module
    #   new_state_dict[name] = v #新字典的key值对应的value一一对应
    model.load_state_dict(model_state['net'])

    # model.eval().cuda()
    model.eval()

    test_dataset = PickleOCFDataset(file_path="/data/hdd/yjn/waymo_data/test_ocf_acc")
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
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


            vehicles_all_flow=data['vehicles_all_flow']
            pedestrians_all_flow=data['pedestrians_all_flow']
            cyclists_all_flow=data['cyclists_all_flow']

          
            vehicles_all_occupancy=data['vehicles_all_occupancy']
            pedestrians_all_occupancy=data['pedestrians_all_occupancy']
            cyclists_all_occupancy=data['cyclists_all_occupancy']
            
            vehicles_future_occluded_occupancy=data['vehicles_future_occluded_occupancy']
            pedestrians_future_occluded_occupancy=data['pedestrians_future_occluded_occupancy']
            cyclists_future_occluded_occupancy=data['cyclists_future_occluded_occupancy']
    
          
            vehicles_future_observed_occupancy=data['vehicles_future_observed_occupancy ']
            pedestrians_future_observed_occupancy=data['pedestrians_future_observed_occupancy ']
            cyclists_future_observed_occupancy=data['cyclists_future_observed_occupancy ']
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
            
            name+=1
            ######
            timestep_grids=create_ground_truth_timestep_grids(
            vehicles_current_occupancy,vehicles_past_occupancy,
            pedestrians_current_occupancy, pedestrians_past_occupancy,
            cyclists_current_occupancy,cyclists_past_occupancy,
            vehicles_all_flow,pedestrians_all_flow,cyclists_all_flow,
            vehicles_all_occupancy,pedestrians_all_occupancy,cyclists_all_occupancy,
            vehicles_future_occluded_occupancy,pedestrians_future_occluded_occupancy,cyclists_future_occluded_occupancy,
            vehicles_future_observed_occupancy,pedestrians_future_observed_occupancy,cyclists_future_observed_occupancy
            
            )

            true_waypoints = occupancy_flow_grids.create_ground_truth_waypoint_grids(
            timestep_grids=timestep_grids, config=config)
            metrics = occupancy_flow_metrics.compute_occupancy_flow_metrics(
            config=config,
            true_waypoints=true_waypoints,
            pred_waypoints=pred_waypoints,
            )
            print(metrics)

if __name__=="__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    inference()