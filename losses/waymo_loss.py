from waymo_open_dataset.protos import occupancy_flow_metrics_pb2
from waymo_open_dataset.utils import occupancy_flow_data
from waymo_open_dataset.utils import occupancy_flow_grids
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


def _occupancy_flow_loss(
        config=None,
        true_waypoints_vehicles_observed_occupancy=None,
        true_waypoints_vehicles_occluded_occupancy=None,
        true_waypoints_vehicles_flow=None,
        pred_waypoint_logits=None,
        device=None
) -> Dict[str, torch.Tensor]:
    """Loss function.

  Args:
    config: OccupancyFlowTaskConfig proto message.
    true_waypoints: Ground truth labels.
    pred_waypoint_logits: Predicted occupancy logits and flows.

  Returns:
    A dict containing different loss tensors:
      observed_xe: Observed occupancy cross-entropy loss.
      occluded_xe: Occluded occupancy cross-entropy loss.
      flow: Flow loss.
  """
    loss_dict = {}
    # Store loss tensors for each waypoint and average at the end.
    loss_dict['observed_xe'] = []
    loss_dict['occluded_xe'] = []
    loss_dict['flow'] = []

    # Iterate over waypoints.
    for k in range(config.num_waypoints):
        # Occupancy cross-entropy loss.
        pred_observed_occupancy_logit = (
            pred_waypoint_logits.vehicles.observed_occupancy[k])
        pred_occluded_occupancy_logit = (
            pred_waypoint_logits.vehicles.occluded_occupancy[k])

        true_observed_occupancy = true_waypoints_vehicles_observed_occupancy[k].to(device).contiguous()

        true_occluded_occupancy = true_waypoints_vehicles_occluded_occupancy[k].to(device).contiguous()
        # Accumulate over waypoints.
        loss_dict['observed_xe'].append(
            _sigmoid_xe_loss(
                true_occupancy=true_observed_occupancy,
                pred_occupancy=pred_observed_occupancy_logit,
                device=device
            )

        )
        loss_dict['occluded_xe'].append(
            _sigmoid_xe_loss(
                true_occupancy=true_occluded_occupancy,
                pred_occupancy=pred_occluded_occupancy_logit,
                device=device
            ))

        # Flow loss.
        pred_flow = pred_waypoint_logits.vehicles.flow[k]
        true_flow = true_waypoints_vehicles_flow[k].to(device).contiguous()
        loss_dict['flow'].append(_flow_loss(true_flow, pred_flow,true_observed_occupancy))

    # Mean over waypoints.
    loss_dict['observed_xe'] = (
            torch.sum(torch.stack(loss_dict['observed_xe'])) / config.num_waypoints)
    loss_dict['occluded_xe'] = (
            torch.sum(torch.stack(loss_dict['occluded_xe'])) / config.num_waypoints)
    loss_dict['flow'] = torch.sum(torch.stack(loss_dict['flow'])) / config.num_waypoints

    return loss_dict


def _sigmoid_xe_loss(
    #TODO: perblems in waymo tutorial 
        pred_occupancy: torch.tensor,
        true_occupancy: torch.Tensor,
        loss_weight: float = 500,
        device=None
) -> torch.tensor:
    """Computes sigmoid cross-entropy loss over all grid cells."""
    # Since the mean over per-pixel cross-entropy values can get very small,
    # we compute the sum and multiply it by the loss weight before computing
    # the mean.
    loss = torch.nn.BCEWithLogitsLoss()
    
    xe_sum = loss(
            input=_batch_flatten(pred_occupancy),
            target=_batch_flatten(true_occupancy),
        )
  
    # Return mean.
    return loss_weight * xe_sum 


def _flow_loss(
        true_flow: torch.tensor,
        pred_flow: torch.tensor,
        true_observed_occupancy:torch.tensor,
        loss_weight: float = 1,
) -> torch.tensor:
    """Computes L1 flow loss."""
  
    diff = true_flow - pred_flow
    # Ignore predictions in areas where ground-truth flow is zero.
    # [batch_size, height, width, 1], [batch_size, height, width, 1]
    true_flow_dx, true_flow_dy = torch.split(true_flow, 1, dim=-1)

    # [batch_size, height, width, 1]
    flow_exists = torch.logical_or(
        torch.ne(true_flow_dx, 0.0),
        torch.ne(true_flow_dy, 0.0),
    )
   
    flow_exists = flow_exists.to(torch.float32)
    diff = diff * flow_exists
    diff_norm = torch.linalg.norm(diff, ord=1, axis=-1)  # L1 norm.
    diff_norm=diff_norm * torch.squeeze(true_observed_occupancy,-1)
    diff_norm_sum = torch.sum(diff_norm)

    flow_exists_sum = torch.sum(flow_exists) / 2

    mean_diff = torch.full_like(diff_norm_sum, fill_value=float('0'))

    mask = (flow_exists_sum != 0)

    mean_diff[mask] = diff_norm_sum[mask] / flow_exists_sum[mask]
    # mean_diff = torch.div(
    #     tf.reduce_sum(diff_norm),
    #     tf.reduce_sum(flow_exists) / 2)  # / 2 since (dx, dy) is counted twice.
    return loss_weight * mean_diff


def _batch_flatten(input_tensor: torch.tensor) -> torch.tensor:
    """Flatten tensor to a shape [batch_size, -1]."""
    return input_tensor.flatten()
