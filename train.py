import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
import torch.nn as nn
from hope import Hope
from utils import OCFDataset
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.optim as optim
from waymo_open_dataset.protos import occupancy_flow_metrics_pb2
from google.protobuf import text_format
from waymo_open_dataset.utils import occupancy_flow_grids
from losses import waymo_loss
from tensorboardX import SummaryWriter
from datetime import datetime

_CURRENT = os.path.abspath(os.path.dirname(__file__))
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
if not os.path.exists(_CURRENT + '/runs'):
        os.mkdir(_CURRENT + '/runs')
writer = SummaryWriter(_CURRENT+'/runs')

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


def train_network():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = Hope()
    model.to(device)
    train_dataset = OCFDataset(file_path="/data/ssd/yjn/waymo_data/train_processed_data/")
    trainloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5,weight_decay=0.01)
    weight_occ=500
    weight_flow=1
    num_epochs = 4
    step=1
    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()
        for data in tqdm(trainloader):
            vehicles_current_occupancy = data['vehicles_current_occupancy']
            vehicles_past_occupancy = data['vehicles_past_occupancy']
            pedestrians_current_occupancy = data['pedestrians_current_occupancy']
            pedestrians_past_occupancy = data['pedestrians_past_occupancy']
            cyclists_current_occupancy = data['cyclists_current_occupancy']
            cyclists_past_occupancy = data['cyclists_past_occupancy']
            vis_grids_roadgraph = data['vis_grids_roadgraph']
            vis_grids_agent_trails = data['vis_grids_agent_trails']
            traffic_grids_trafficgraph = data['traffic_grids_trafficgraph']
            true_waypoints_vehicles_observed_occupancy = data['true_waypoints_vehicles_observed_occupancy']
            true_waypoints_vehicles_occluded_occupancy = data[' true_waypoints_vehicles_occluded_occupancy']
            true_waypoints_vehicles_flow = data['true_waypoints_vehicles_flow']
            model_inputs = torch.cat(
                (
                    vehicles_current_occupancy,
                    vehicles_past_occupancy,
                    pedestrians_current_occupancy,
                    pedestrians_past_occupancy,
                    cyclists_current_occupancy,
                    cyclists_past_occupancy,
                    vis_grids_roadgraph,
                    traffic_grids_trafficgraph
                ),
                dim=-1,
            )
            model_inputs = model_inputs.permute(0, 3, 1, 2).contiguous().to(device)
            model_outputs = model(model_inputs)
            # print(model_outputs.shape)
            pred_waypoint_logits = _get_pred_waypoint_logits(model_outputs)
           # print(pred_waypoint_logits)
            # Compute loss.
            optimizer.zero_grad()
            loss_dict = waymo_loss._occupancy_flow_loss(
                config=config,
                true_waypoints_vehicles_observed_occupancy=true_waypoints_vehicles_observed_occupancy,
                true_waypoints_vehicles_occluded_occupancy=true_waypoints_vehicles_occluded_occupancy,
                true_waypoints_vehicles_flow=true_waypoints_vehicles_flow,
                pred_waypoint_logits=pred_waypoint_logits,
                device=device)
            observed_loss=loss_dict['observed_xe']
            occluded_loss=loss_dict['occluded_xe']
            flow_loss=loss_dict['flow']
            loss=weight_occ*(observed_loss+occluded_loss)+weight_flow*flow_loss
            loss.backward()
            optimizer.step()
            writer.add_scalar("total_loss",loss,step)
            writer.add_scalar("oberved_loss",observed_loss,step)
            writer.add_scalar("occluded_loss",occluded_loss,step)
            writer.add_scalar("flow_loss",flow_loss,step)
            step+=1
            print('Epoch: {}/{} \tTraining Loss: {:.6f}'.format(epoch + 1, num_epochs,loss))
        if not os.path.exists(_CURRENT + '/models'):
            os.mkdir(_CURRENT + '/models')

        timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        PATH = _CURRENT + '/models/smallRegularizedCNN_L%.3f_%s.pth' % (loss, timestamp)
        torch.save(model.state_dict(), PATH)
    writer.close()
    print('Finished Training')







if __name__ == '__main__':
    train_network()
