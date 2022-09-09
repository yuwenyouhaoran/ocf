import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import numpy as np
import tensorflow as tf
import torch.nn
from google.protobuf import text_format
from tqdm import tqdm
from waymo_open_dataset.protos import occupancy_flow_metrics_pb2
from waymo_open_dataset.utils import occupancy_flow_data
from waymo_open_dataset.utils import occupancy_flow_grids
from joblib import Parallel, delayed,parallel_backend
import occupancy_flow_grids_new
from sample import PickleDatabase


def create_dataset(datapath):
    # files = os.listdir(datapath)
    # dataset = tf.data.TFRecordDataset(
    #     [os.path.join(datapath, f) for f in files], num_parallel_reads=1
    # )
    # if n_shards > 1:
    #     dataset = dataset.shard(n_shards, shard_id)
    # return dataset
    filenames = tf.io.matching_files(datapath)
    print(filenames)
    for i in filenames:
        print(i)

    dataset = tf.data.TFRecordDataset(filenames)
    # dataset = dataset.repeat(1)

    dataset = dataset.map(occupancy_flow_data.parse_tf_example)
    dataset = dataset.batch(1)

    return dataset


def data_to_numpy(data):
    for k, v in data.items():
        data[k] = v.numpy()


def save_dict_by_numpy(filename, dict_vale):
    if not (os.path.exists(os.path.dirname(filename))):
        os.mkdir(os.path.dirname(filename))
    np.save(filename, dict_vale)


def dataset_process(i, out_dir, config):
    #data_dir = os.path.join(out_dir, str(i.numpy(), 'UTF-8').split('/')[-1])
    #if not os.path.exists(data_dir):
        #os.mkdir(data_dir)
    dataset = tf.data.TFRecordDataset(i)
    # dataset = dataset.repeat(1)
    dataset = dataset.map(occupancy_flow_data.parse_tf_example)
    dataset = dataset.batch(1)
   # database = PickleDatabase(database_path=data_dir, write=True)
    scene_number = 0
    for data in tqdm(iter(dataset)):
        scene_number += 1
        inputs = occupancy_flow_data.add_sdc_fields(data)
        scenario_id = str(inputs["scenario/id"].numpy()[0], "ascii")

        timestep_grids_origin = occupancy_flow_grids.create_ground_truth_timestep_grids(
            inputs=inputs, config=config)

        true_waypoints = occupancy_flow_grids.create_ground_truth_waypoint_grids(
            timestep_grids=timestep_grids_origin, config=config)
        for i in range(config.num_waypoints):
            true_waypoints.vehicles.observed_occupancy[i] = tf.squeeze(true_waypoints.vehicles.observed_occupancy[i],
                                                                       axis=0).numpy()
            true_waypoints.vehicles.occluded_occupancy[i] = tf.squeeze(true_waypoints.vehicles.occluded_occupancy[i],
                                                                       axis=0).numpy()
            true_waypoints.vehicles.flow[i] = tf.squeeze(true_waypoints.vehicles.flow[i], axis=0).numpy()

        vis_grids = occupancy_flow_grids_new.create_ground_truth_vis_grids(
            inputs=inputs, timestep_grids=timestep_grids_origin, config=config)

        timestep_grids_new = occupancy_flow_grids_new.create_ground_truth_timestep_grids(
            inputs=inputs, config=config)

        traffic_grids = occupancy_flow_grids_new.create_ground_truth_traffic_grids(inputs=inputs, config=config)

        data_new = {
            "vehicles_current_occupancy": tf.squeeze(timestep_grids_new.vehicles.current_occupancy, axis=0).numpy(),
            "vehicles_past_occupancy": tf.squeeze(timestep_grids_new.vehicles.past_occupancy, axis=0).numpy(),
            "pedestrians_current_occupancy": tf.squeeze(timestep_grids_new.pedestrians.current_occupancy,
                                                        axis=0).numpy(),
            "pedestrians_past_occupancy": tf.squeeze(timestep_grids_new.pedestrians.past_occupancy, axis=0).numpy(),
            "cyclists_current_occupancy": tf.squeeze(timestep_grids_new.cyclists.current_occupancy, axis=0).numpy(),
            "cyclists_past_occupancy": tf.squeeze(timestep_grids_new.cyclists.past_occupancy, axis=0).numpy(),
            "vis_grids_roadgraph": tf.squeeze(vis_grids.roadgraph, axis=0).numpy(),
            "vis_grids_agent_trails": tf.squeeze(vis_grids.agent_trails, axis=0).numpy(),
            "traffic_grids_trafficgraph": tf.squeeze(traffic_grids.trafficgraph).numpy(),
            "true_waypoints_vehicles_observed_occupancy": true_waypoints.vehicles.observed_occupancy,
            " true_waypoints_vehicles_occluded_occupancy": true_waypoints.vehicles.occluded_occupancy,
            "true_waypoints_vehicles_flow": true_waypoints.vehicles.flow
        }
        output_filename = os.path.join(out_dir, "{}".format(scenario_id))
        save_dict_by_numpy(output_filename,data_new)
       # key = scenario_id
       # database.put(key, data_new)


def data_process():
    # A tfrecord containing tf.Example protos as downloaded from the Waymo Open
    # Dataset (motion) webpage.

    # Replace this path with your own tfrecords.
    # DATASET_FOLDER = '/path/to/waymo_open_dataset_motion_v_1_1_0/uncompressed'
    DATASET_FOLDER ='/data/ssd/yjn/waymo_data'
    #TRAIN_FILES = f'{DATASET_FOLDER}/tf_example/training/training_tfexample.tfrecord*'
    TRAIN_FILES = f'{DATASET_FOLDER}/tf_example/training'
    out_dir = '/data/ssd/yjn/waymo_data/train_processed_data'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
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
    train_datasets_start = ["00977"]
    train_datasets_end=["01000"]
    train_list=['training_tfexample.tfrecord-{}-of-{}'.format(train_datasets_start[i],train_datasets_end[0]) for i in range(len(train_datasets_start))]

    files=[os.path.join(TRAIN_FILES,train_list[i])for i in range(len(train_list))]

    filenames = tf.io.matching_files(files)
    print(filenames)
    with parallel_backend('threading', n_jobs=10):
        Parallel()(delayed(dataset_process)(i, out_dir, config)
                       for i in filenames)
    # for i in filenames:
    #
    #     data_dir=os.path.join(out_dir,str(i.numpy(),'UTF-8').split('/')[-1])
    #     if not os.path.exists(data_dir):
    #         os.mkdir(data_dir)
    #     dataset = tf.data.TFRecordDataset(i)
    # # dataset = dataset.repeat(1)
    #     dataset = dataset.map(occupancy_flow_data.parse_tf_example)
    #     dataset = dataset.batch(1)
    #     database = PickleDatabase(database_path=data_dir, write=True)
    #     scene_number=0
    #     for data in tqdm(iter(dataset)):
    #         scene_number+=1
    #         inputs = occupancy_flow_data.add_sdc_fields(data)
    #         scenario_id=str(inputs["scenario/id"].numpy()[0],"ascii")
    #
    #
    #         timestep_grids_origin = occupancy_flow_grids.create_ground_truth_timestep_grids(
    #         inputs=inputs, config=config)
    #
    #         true_waypoints = occupancy_flow_grids.create_ground_truth_waypoint_grids(
    #         timestep_grids=timestep_grids_origin, config=config)
    #         for i in range(config.num_waypoints):
    #             true_waypoints.vehicles.observed_occupancy[i] = tf.squeeze(true_waypoints.vehicles.observed_occupancy[i],
    #                                                                    axis=0).numpy()
    #             true_waypoints.vehicles.occluded_occupancy[i] = tf.squeeze(true_waypoints.vehicles.occluded_occupancy[i],
    #                                                                    axis=0).numpy()
    #             true_waypoints.vehicles.flow[i] = tf.squeeze(true_waypoints.vehicles.flow[i], axis=0).numpy()
    #
    #         vis_grids = occupancy_flow_grids_new.create_ground_truth_vis_grids(
    #         inputs=inputs, timestep_grids=timestep_grids_origin, config=config)
    #
    #         timestep_grids_new = occupancy_flow_grids_new.create_ground_truth_timestep_grids(
    #         inputs=inputs, config=config)
    #
    #         traffic_grids = occupancy_flow_grids_new.create_ground_truth_traffic_grids(inputs=inputs, config=config)
    #
    #         data_new = {
    #         "vehicles_current_occupancy": tf.squeeze(timestep_grids_new.vehicles.current_occupancy, axis=0).numpy(),
    #         "vehicles_past_occupancy": tf.squeeze(timestep_grids_new.vehicles.past_occupancy, axis=0).numpy(),
    #         "pedestrians_current_occupancy": tf.squeeze(timestep_grids_new.pedestrians.current_occupancy,
    #                                                     axis=0).numpy(),
    #         "pedestrians_past_occupancy": tf.squeeze(timestep_grids_new.pedestrians.past_occupancy, axis=0).numpy(),
    #         "cyclists_current_occupancy": tf.squeeze(timestep_grids_new.cyclists.current_occupancy, axis=0).numpy(),
    #         "cyclists_past_occupancy": tf.squeeze(timestep_grids_new.cyclists.past_occupancy, axis=0).numpy(),
    #         "vis_grids_roadgraph": tf.squeeze(vis_grids.roadgraph, axis=0).numpy(),
    #         "vis_grids_agent_trails": tf.squeeze(vis_grids.agent_trails, axis=0).numpy(),
    #         "traffic_grids_trafficgraph": tf.squeeze(traffic_grids.trafficgraph).numpy(),
    #         "true_waypoints_vehicles_observed_occupancy": true_waypoints.vehicles.observed_occupancy,
    #         " true_waypoints_vehicles_occluded_occupancy": true_waypoints.vehicles.occluded_occupancy,
    #         "true_waypoints_vehicles_flow": true_waypoints.vehicles.flow
    #         }
    #         # output_filename = os.path.join(data_dir, "{}".format(scenario_id))
    #         # save_dict_by_numpy(output_filename,data_new)
    #         key=scenario_id
    #         database.put(key,data_new)


if __name__ == '__main__':
    data_process()
    # print(torch.nn.Embedding(1,2).shape)
