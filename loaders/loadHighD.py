import pandas

from src.data_management.read_csv import *


class DataGetter:

    def __init__(self, path, index):
        str_index = str(index).zfill(2)
        self._data = pandas.read_csv(path + str_index + '_tracks.csv')
        self._data_meta = pandas.read_csv(path + str_index + '_tracksMeta.csv')
        self._car_id_list = self._data_meta.groupby(CLASS).get_group('Car')[ID].values
        self._recordingMeta = read_meta_info({'input_meta_path': path + str_index + '_recordingMeta.csv'})
        self._id_around = [PRECEDING_ID, FOLLOWING_ID,
                           LEFT_PRECEDING_ID, LEFT_ALONGSIDE_ID,  LEFT_FOLLOWING_ID,
                           RIGHT_PRECEDING_ID, RIGHT_ALONGSIDE_ID, RIGHT_FOLLOWING_ID]
        self._lane_ids = np.sort(self._data[LANE_ID].unique())
        self._lane_dirs = self._get_lane_dirs()
        self.data_to_get = [X, Y, X_VELOCITY, Y_VELOCITY, X_ACCELERATION, Y_ACCELERATION]

    def _get_lane_dirs(self):
        lane_dirs = []
        for id in self._lane_ids:
            data_track = self._data[self._data[LANE_ID] == id]
            track_id = data_track[ID]
            meta_track = self._data_meta[self._data_meta[TRACK_ID] == track_id.values[0]]
            direction = meta_track[DRIVING_DIRECTION]
            direction = direction.values[0]
            lane_dirs.append(direction)
        return lane_dirs

    def _time2frame(self, time):
        frame = time * self._recordingMeta[FRAME_RATE]
        return np.rint(frame)

    def _frame2time(self, frame):
        return frame/self._recordingMeta[FRAME_RATE]

    def get_time_list(self):
        return self._data[FRAME].values/self._recordingMeta[FRAME_RATE]

    def get_data_at_time(self, time):
        frame = self._time2frame(time)
        mask_at_time = self._data[FRAME] == int(frame)
        return self._data[mask_at_time]

    def get_observations_times(self, vehicle_id):
        return self._frame2time(self._data[self._data[ID == vehicle_id]][FRAME]).values

    def lane_id_to_index(self, lane_id):
        if isinstance(lane_id, int) or isinstance(lane_id, np.integer):
            return np.squeeze(np.argwhere(self._lane_ids == lane_id))
        else:
            return np.argwhere(lane_id[:, np.newaxis] == self._lane_ids)[:, 1]

    def get_neighbor_lane_index(self, lane_id):
        lane_index = self.lane_id_to_index(lane_id)
        lane_direction = self._lane_dirs[lane_index]
        neighbor_lanes = [int(lane_index)]
        if lane_index > 0 and self._lane_dirs[lane_index-1] == lane_direction:
            neighbor_lanes.append(int(lane_index-1))
        else:
            neighbor_lanes.append(None)
        if lane_index < len(self._lane_ids)-1 and self._lane_dirs[lane_index+1] == lane_direction:
            neighbor_lanes.append(int(lane_index+1))
        else:
            neighbor_lanes.append(None)
        if lane_direction == 1:
            neighbor_lane_temp = neighbor_lanes[1]
            neighbor_lanes[1] = neighbor_lanes[2]
            neighbor_lanes[2] = neighbor_lane_temp
        return neighbor_lanes

    # Returns True for left to right direction and False otherwise
    def get_lane_direction(self, lane_id):
        lane_index = self.lane_id_to_index(lane_id)
        return self._lane_dirs[lane_index] == 2

    def get_data_near(self, time, position):
        data_at_time = self.get_data_at_time(time)
        data_at_time_n_pos = data_at_time[data_at_time[X] == position[0]]
        data_at_time_n_pos = data_at_time_n_pos[data_at_time_n_pos[Y] == position[1]]
        id_around = data_at_time_n_pos[self._id_around].values[0]
        data_near = []
        for id in id_around:
            if id > 0:
                data_near.append(data_at_time[data_at_time[ID] == id])
            else:
                data_near.append(None)
        return data_near

    def get_data_around(self, time, id):
        data_at_time = self.get_data_at_time(time)
        data_at_time_n_id = data_at_time[data_at_time[ID] == id]
        id_around = data_at_time_n_id[self._id_around].values[0]
        data_near = []
        for id in id_around:
            if id > 0:
                data_near.append(data_at_time[data_at_time[ID] == id])
            else:
                data_near.append(None)
        return data_near

    def get_id(self, position):
        data_at_pos = self._data[self._data[X] == position[0]]
        data_at_pos = data_at_pos[data_at_pos[Y] == position[1]]
        return data_at_pos[ID].values[0]

    # not tested
    def get_position(self, time, id):
        data_id = self._data[self._data[TRACK_ID] == id]
        data_id_n_time = data_id[data_id[FRAME] == self._time2frame(time)]
        pos = data_id_n_time[[X, Y]].values[0]
        return pos

    def get_example(self, size_limit=100, ego_id=None):
        if ego_id is None:
            is_empty = True
            while is_empty:
                chosen_id = np.random.choice(self._car_id_list)
                data_ego = self._data[self._data[TRACK_ID] == chosen_id]
                # print(data_ego.shape)
                is_empty = data_ego.shape[0] < size_limit
        else:
            chosen_id = ego_id
            data_ego = self._data[self._data[TRACK_ID] == chosen_id]
            is_empty = data_ego.shape[0] < size_limit
            if is_empty:
                print('Warning: given id refers to a track with less than ', size_limit, ' observations.')
        return self.get_example_id(chosen_id, data_ego)

    def get_all_examples(self, min_size_limit=100):
        all_examples = []
        for id in self._car_id_list:
            data_ego = self._data[self._data[TRACK_ID] == id]
            is_empty = data_ego.shape[0] < min_size_limit
            if not is_empty:
                all_examples.append(self.get_example_id(id, data_ego))
        return all_examples

    def get_example_id(self, ego_id, data_ego):
        time_array = self._frame2time(data_ego[FRAME].values)
        example = []
        for time in time_array:
            example.append(self.get_data_around(time, ego_id))
        return example, data_ego

    def get_numpy_example(self, example=None):
        if example is None:
            example_around, example_ego = self.get_example()
        else:
            example_around, example_ego = example
        example_out = np.zeros(shape=[example_ego.shape[0], len(example_around[0])+1, len(self.data_to_get)])

        ego_lane = self.lane_id_to_index(example_ego[LANE_ID].values)
        ego_id = example_ego[TRACK_ID].values[0]
        ego = example_ego[self.data_to_get].values
        example_out[:, 0, :] = ego
        for i, one_around in enumerate(example_around): # Time
            for j, one_one in enumerate(one_around): # Neighbors
                if one_one is not None:
                    surroundings = one_one[self.data_to_get].values
                else:
                    surroundings = np.zeros([len(self.data_to_get)])

                example_out[i, j, :] = surroundings

        return example_out, ego_lane, ego_id

    def get_old_style_example(self, sequence_length):
        size = 0
        while 3*size < sequence_length:
            example, ego_lane, ego_id = self.get_numpy_example()
            size = example.shape[0]
        old_example_past = np.zeros([example.shape[0] - 2 * sequence_length,
                                     sequence_length, example.shape[1], example.shape[2]])
        old_example_pred = np.zeros([example.shape[0] - 2 * sequence_length,
                                     sequence_length, example.shape[1], example.shape[2]])

        for i in range(sequence_length):

            if i < sequence_length:
                old_example_past[:, i, :, :] = example[i:i - 2 * sequence_length, :, :]
                old_example_pred[:, i, :, :] = example[i + sequence_length:i - sequence_length, :, :]
            else:
                old_example_past[:, i, :, :] = example[sequence_length:-sequence_length, :, :]
                old_example_pred[:, i, :, :] = example[2 * sequence_length:, :, :]

        return old_example_past, old_example_pred, ego_lane, ego_id

