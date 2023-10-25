# Setup

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import datetime, os
import gc

import scipy.stats as ss
import json

from transformers import InformerConfig, InformerForPrediction

from gluonts.time_feature import TimeFeature
from gluonts.dataset.field_names import FieldName
from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    RemoveFields,
    SelectFields,
    SetField,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
    VstackFeatures,
    RenameFields,
)

from transformers import PretrainedConfig

from gluonts.transform.sampler import InstanceSampler
from typing import Optional

from typing import Iterable

from gluonts.itertools import Cached, Cyclic
from gluonts.dataset.loader import as_stacked_batches

from accelerate import Accelerator
from torch.optim import AdamW

import argparse

class ConvLSTMNet(nn.Module):
    def __init__(self, input_dim, nb_feature, hidden_dim, num_layers, output_dim, forward_steps, drop_prob=0.):
        super(ConvLSTMNet, self).__init__()

        # Convolutional layers
        self.cnn = nn.Sequential(
        nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
        nn.ReLU(),
        )

        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)

        # LSTM layers
        self.lstm = nn.LSTM(input_size=nb_feature, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)

        # Add dropout
        self.dropout = nn.Dropout(drop_prob)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # Bidirectional LSTM doubles the hidden dimension

        self.forward_steps = forward_steps

    def forward(self, x):
        # Convolutional layers

        x = self.cnn(x)

        # LSTM layers
        lstm_out, _ = self.lstm(x)

        last_lstm = lstm_out[:, -self.forward_steps:, :]

        # Fully connected layer
        out = self.fc(last_lstm)  # We use the output of the last time steps according to forward steps

        out = nn.functional.softmax(out,dim=-1)

        out = out.transpose(-2,-1)

        return out

    def sample(self, x):
      output = self(x)

      sample = torch.zeros_like(output)

      for score_idx, score in enumerate(output):

        law = score.transpose(0,1).squeeze()
        dist = torch.distributions.categorical.Categorical(law)
        action = dist.sample().item()

        sample[score_idx,action,0] = 1.

      return sample


def progress_array(action_length):
  return np.ndarray.tolist(np.linspace(0, 1, num=action_length))

def create_transformation(config: PretrainedConfig) -> Transformation:
    # create list of fields to remove later
    remove_field_names = []
    if config.num_static_real_features == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_REAL)
    if config.num_dynamic_real_features == 0:
        remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)
    if config.num_static_categorical_features == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_CAT)

    return Chain(
        # step 1: remove static/dynamic fields if not specified
        [RemoveFields(field_names=remove_field_names)]
        # step 2: convert the data to NumPy (potentially not needed)
        + (
            [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_CAT,
                    expected_ndim=1,
                    dtype=int,
                )
            ]
            if config.num_static_categorical_features > 0
            else []
        )
        + (
            [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_REAL,
                    expected_ndim=1,
                )
            ]
            if config.num_static_real_features > 0
            else []
        )
        + [
            AsNumpyArray(
                field=FieldName.TARGET,
                # we expect an extra dim for the multivariate case:
                expected_ndim=1 if config.input_size == 1 else 2,
            ),
            # step 3: handle the NaN's by filling in the target with zero
            # and return the mask (which is in the observed values)
            # true for observed values, false for nan's
            # the decoder uses this mask (no loss is incurred for unobserved values)
            # see loss_weights inside the xxxForPrediction model
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            ),
            # step 5: add another temporal feature (just a single number)
            # tells the model where in the life the value of the time series is
            # sort of running counter
            AddAgeFeature(
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_AGE,
                pred_length=config.prediction_length,
                log_scale=True,
            ),
            # step 6: vertically stack all the temporal features into the key FEAT_TIME
            VstackFeatures(
                output_field=FieldName.FEAT_TIME,
                input_fields=[FieldName.FEAT_AGE]
                + (
                    [FieldName.FEAT_DYNAMIC_REAL]
                    if config.num_dynamic_real_features > 0
                    else []
                ),
            ),
            # step 7: rename to match HuggingFace names
            RenameFields(
                mapping={
                    FieldName.FEAT_STATIC_CAT: "static_categorical_features",
                    FieldName.FEAT_STATIC_REAL: "static_real_features",
                    FieldName.FEAT_DYNAMIC_REAL: "dynamic_real_features",
                    FieldName.FEAT_TIME: "time_features",
                    FieldName.TARGET: "values",
                    FieldName.OBSERVED_VALUES: "observed_mask",
                }
            ),
        ]
    )

def create_instance_splitter(
    config: PretrainedConfig,
    mode: str,
    train_sampler: Optional[InstanceSampler] = None,
    validation_sampler: Optional[InstanceSampler] = None,
) -> Transformation:
    assert mode in ["train", "validation", "test"]

    instance_sampler = {
        "train": train_sampler
        or ExpectedNumInstanceSampler(
            num_instances=1.0, min_future=config.prediction_length
        ),
        "validation": validation_sampler
        or ValidationSplitSampler(min_future=config.prediction_length),
        "test": TestSplitSampler(),
    }[mode]

    return InstanceSplitter(
        target_field="values",
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=instance_sampler,
        past_length=config.context_length + max(config.lags_sequence),
        future_length=config.prediction_length,
        time_series_fields=["time_features", "observed_mask"],
    )

def create_train_dataloader(
    config: PretrainedConfig,
    data,
    batch_size: int,
    num_batches_per_epoch: int,
    shuffle_buffer_length: Optional[int] = None,
    cache_data: bool = True,
    **kwargs,
) -> Iterable:
    PREDICTION_INPUT_NAMES = [
        "past_time_features",
        "past_values",
        "past_observed_mask",
        "future_time_features",
    ]
    if config.num_static_categorical_features > 0:
        PREDICTION_INPUT_NAMES.append("static_categorical_features")

    if config.num_static_real_features > 0:
        PREDICTION_INPUT_NAMES.append("static_real_features")

    TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
        "future_values",
        "future_observed_mask",
    ]

    transformation = create_transformation(config)
    transformed_data = transformation.apply(data, is_train=True)

    if cache_data:
        transformed_data = Cached(transformed_data)

    # we initialize a Training instance
    instance_splitter = create_instance_splitter(config, "train")

    # the instance splitter will sample a window of
    # context length + lags + prediction length (from all the possible transformed time series, 1 in our case)
    # randomly from within the target time series and return an iterator.
    stream = Cyclic(transformed_data).stream()
    training_instances = instance_splitter.apply(
        stream, is_train=True
    )

    return as_stacked_batches(
        training_instances,
        batch_size=batch_size,
        shuffle_buffer_length=shuffle_buffer_length,
        field_names=TRAINING_INPUT_NAMES,
        output_type=torch.tensor,
        num_batches_per_epoch=num_batches_per_epoch,
    )

def create_test_dataloader(
    config: PretrainedConfig,
    data,
    batch_size: int,
    **kwargs,
):
    PREDICTION_INPUT_NAMES = [
        "past_time_features",
        "past_values",
        "past_observed_mask",
        "future_time_features",
    ]
    if config.num_static_categorical_features > 0:
        PREDICTION_INPUT_NAMES.append("static_categorical_features")

    if config.num_static_real_features > 0:
        PREDICTION_INPUT_NAMES.append("static_real_features")

    transformation = create_transformation(config)
    transformed_data = transformation.apply(data, is_train=False)

    # we create a Test Instance splitter which will sample the very last
    # context window seen during training only for the encoder.
    instance_sampler = create_instance_splitter(config, "test")

    # we apply the transformations in test mode
    testing_instances = instance_sampler.apply(transformed_data, is_train=False)

    return as_stacked_batches(
        testing_instances,
        batch_size=batch_size,
        output_type=torch.tensor,
        field_names=PREDICTION_INPUT_NAMES,
    )

def outliers_idx(y):
  idx = []
  keep = []
  high_occ = []

  props = torch.sum(y,0)
  tot = props.transpose(0,1).squeeze().sum()
  mean = tot / props.shape[0]

  # FOR TESTING !!!
  mean = 79

  # Target over occurent indexes.
  for i in range(props.shape[0]):
    if props[i,0] > mean:
      idx.append(i)

      # Exclude them for low occurence maximum computation
      high_occ.append(props[i,0].item())
      props[i,0] = 0.


  low_max = torch.max(props).item()

  # High occurent label should appear in average as usual as
  #the most occurent label in low occurent label.
  mean_occ = low_max
  for i in range(len(idx)):
    keep_tensor = mean_occ / high_occ[i]
    keep.append(keep_tensor)


  return idx, keep

def to_remove_bias(y, outliers, keep):
  cond = torch.ones(y.shape[0])
  # Do the process for each outliers indexes
  for idx in range(len(outliers)):
    out_idx = outliers[idx]
    keep_val = keep[idx] # Proportion to keep

    # For each row of the dataset, if the action is the outliers
    # keep the row with probability kee_val
    for row in range(y.shape[0]):
      curr_action = None
      for action_idx, action in enumerate(y[row]):
        if action[0] == 1.:
          curr_action = action_idx

      if curr_action == out_idx:
        rand = torch.rand(1).item()
        if rand > keep_val:
          cond[row] = 0

  return cond.int()

def simulation(action_model, norm_model, duration, init_state,device):
  norm_model.eval()
  action_model.eval()

  # Convert duration in minutes into max step at 50Hz
  max_steps = int(duration * 60 * 50)
  print("max_steps")
  print(max_steps)
  step = 0

  action_batch, norm_batch = init_state[0], init_state[1]
  action_batch = action_batch.to(device)

  nb_future = norm_batch["future_time_features"].shape[1]
  nb_max_future = norm_batch["future_time_features"].shape[1]
  nb_past = norm_batch["past_values"].shape[1]

  forecasts_ = []
  forecasts = torch.zeros((1,2))
  labels = np.zeros((1,8))

  while step < max_steps:

    # Forecast action
    next_action = action_model.sample(action_batch)

    next_action_vec = next_action.transpose(1,2)
    labels = np.concatenate((labels,next_action_vec.squeeze(0).cpu().numpy()), axis=0)

    action_batch = torch.cat([action_batch[0,1:,:].unsqueeze(0),next_action_vec],dim=1)

    future_action = next_action.squeeze(0).transpose(0,1)
    action_feature = torch.cat(nb_max_future * [future_action],dim=0)
    norm_batch["future_time_features"][0,:,1:9] = action_feature

    # Forecast norms
    outputs = norm_model.generate(
        past_time_features=norm_batch["past_time_features"].to(device),
        past_values=norm_batch["past_values"].to(device),
        future_time_features=norm_batch["future_time_features"].to(device),
        past_observed_mask=norm_batch["past_observed_mask"].to(device),
    )
    forecasts_.append(outputs.sequences.cpu().numpy())

    for prog_idx, prog in enumerate(forecasts_[-1][0,0,:,-1]):
      if prog > 0.85:
        break
    nb_future = prog_idx+1
    forecasts_[-1] = forecasts_[-1][:,:,:nb_future,:]
    forecasts_[-1][0,0,:,-1] = np.linspace(start=-1.,stop=1.,num=nb_future)
    forecasts_[-1][0,0,-1,-1] = -1

    save_past_time = torch.clone(norm_batch["past_time_features"][0,nb_future:,:])
    norm_batch["past_time_features"][0,:nb_past - nb_future,:] = save_past_time
    norm_batch["past_time_features"][0,nb_past - nb_future:,:] = norm_batch["future_time_features"][0,:nb_future]

    save_past_values = torch.clone(norm_batch["past_values"][0,nb_future:,:])
    norm_batch["past_values"][0,:nb_past - nb_future,:] = save_past_values


    norm_batch["past_values"][0,nb_past - nb_future:,:] = torch.from_numpy(forecasts_[-1][0,0,:,:])

    save_future_time = norm_batch["future_time_features"][0,nb_future:,0]
    norm_batch["future_time_features"][0,:nb_max_future-nb_future,0] = save_future_time
    for i in range(nb_max_future-nb_future,nb_max_future):
      norm_batch["future_time_features"][0,i,0] = norm_batch["future_time_features"][0,i-1,0] + 1e-4

    forecasts = np.concatenate((forecasts,forecasts_[-1][0,0]),axis=0)

    step += nb_future
    print(f"Gait steps: {nb_future}")
    print(f"Total steps: {step}")

  mean = np.mean(forecasts,axis=0)

  for val in forecasts:
    if val[-2] < -1.:
      val[-2] = mean[-2]

  forecasts[0,-1] = -1

  return forecasts, labels[1:]

def train(path=os.getcwd()):

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  match_1_df = pd.read_json(f"{path}/match_1.json",
     convert_dates=True)

  match_2_df = pd.read_json(f"{path}/match_2.json",
     convert_dates=True)

  match_df = pd.concat([match_1_df.copy(), match_2_df.copy()],axis=0).reset_index(drop=True)
  
  # Data preprocessing

  ## Manage "no action" label
  match_df.replace('no action', 'walk', inplace=True)

  ## One hot action encoder
  progress_column = match_df["norm"].apply(len).apply(progress_array)
  match_df.insert(len(match_df.columns), 'progress', progress_column.values)
  match_df = pd.get_dummies(match_df, columns=['label', ])

  ## Action forecasting

  ### Action history
  actions_col = []
  for col in match_df.columns:
    if "label_" in col:
      actions_col.append(col)

  actions_df = match_df[actions_col]

  ### Lag + multi step
  lag = 3
  forward = 1
  actions_lag_df = pd.concat([actions_df.shift(i).add_suffix(" - "+str(i)) for i in range(lag,0,-1)]
                             + [actions_df.shift(-i).add_suffix(" + "+str(i)) for i in range(0,forward,1)], axis=1)
  action_column_nb = len(actions_lag_df.columns)

  ## Norm forecasting

  ### One row per norm
  continous_df = pd.DataFrame(columns=match_df.columns)

  for row_ind in match_df.index:
    duration = len(match_df["norm"][row_ind])
    for step in range(duration):
      new_row = dict()
      for column_name in match_df.columns:
        if "label_" in column_name:
          new_row[column_name] = match_df[column_name][row_ind]
        else:
          new_row[column_name] = match_df[column_name][row_ind][step]
      continous_df.loc[len(continous_df)] = new_row

  #continous_df.to_csv(f'{path}/continous_df.csv', index=False)

  ### Place predicted features at the end of the dataframe.
  nb_features = len(continous_df.columns)
  new_cols = ['label_cross', 'label_dribble', 'label_pass', 'label_rest', 'label_run', 'label_shot', 'label_tackle', 'label_walk','norm', 'progress']
  continous_df=continous_df[new_cols]

  ### Scale data
  train_prop = 0.8

  raw_values = continous_df[["norm","progress"]].values

  # Perform scaling only on the train split to avoid information leakage
  train_raw_values = raw_values[:int(0.8 * raw_values.shape[0])]
  scaler_train = MinMaxScaler(feature_range=(-1, 1))
  scaled_train = scaler_train.fit_transform(train_raw_values)
  scaler_test = MinMaxScaler(feature_range=(-1, 1))
  scaled_test = scaler_test.fit_transform(raw_values[int(0.8 * raw_values.shape[0]):])

  scaled = np.concatenate((scaled_train,scaled_test),axis=0)

  continous_df_scaled = continous_df.copy()
  continous_df_scaled["norm"] = scaled[:,0]
  continous_df_scaled["progress"] = scaled[:,1]

  # Norm forecasting

  ## Informer

  ### Dataset
  prediction_length = 100
  scaled_numpy = np.transpose(continous_df_scaled.to_numpy())
  scaled_numpy = np.float32(scaled_numpy)
  train_idx = int(scaled_numpy.shape[-1] * train_prop)

  multi_variate_train_dataset = [{
      'feat_dynamic_real': scaled_numpy[:-2,:train_idx],
      'target': scaled_numpy[-2:,:train_idx],
      'start': 0,
  }]

  multi_variate_test_dataset = [{
      'feat_dynamic_real': scaled_numpy[:-2,train_idx:],
      'target': scaled_numpy[-2:,train_idx:],
      'start': 0,
  }]

  # Take into account prediction length
  multi_variate_test_dataset[0]['target'] = multi_variate_test_dataset[0]['target'][:,prediction_length:]

  lags_sequence =  [1, 2, 3]

  ### Model
  config = InformerConfig(
      # in the multivariate setting, input_size is the number of variates in the time series per time step
      input_size=multi_variate_train_dataset[0]['target'].shape[0],
      # prediction length:
      prediction_length=prediction_length,
      # context length:
      context_length=prediction_length * 2,
      # lags value copied from 1 week before:
      lags_sequence=lags_sequence,
      # we'll add 5 time features ("hour_of_day", ..., and "age"):
      num_time_features=1,
      # Dynamic real features dim
      num_dynamic_real_features = multi_variate_train_dataset[0]['feat_dynamic_real'].shape[0],
      num_parallel_samples = 1,

      # informer params:
      dropout=0.1,
      encoder_layers=6,
      decoder_layers=4,
      # project input from num_of_variates*len(lags_sequence)+num_time_features to:
      d_model=64,
  )

  model = InformerForPrediction(config)

  ### Data transformations

  ### Instance splitter

  ### Dataloader

  train_dataloader = create_train_dataloader(
      config=config,
      data=multi_variate_train_dataset,
      batch_size=256,
      num_batches_per_epoch=100,
      num_workers=2,
  )

  test_dataloader = create_test_dataloader(
      config=config,
      data=multi_variate_test_dataset,
      batch_size=32,
  )

  ### Train

  epochs = 2
  loss_history = []

  accelerator = Accelerator()
  device = accelerator.device

  model.to(device)
  optimizer = AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=1e-1)

  model, optimizer, train_dataloader = accelerator.prepare(
      model,
      optimizer,
      train_dataloader,
  )

  model.train()
  for epoch in range(epochs):
      for idx, batch in enumerate(train_dataloader):
          optimizer.zero_grad()
          outputs = model(
              static_categorical_features=batch["static_categorical_features"].to(device)
              if config.num_static_categorical_features > 0
              else None,
              static_real_features=batch["static_real_features"].to(device)
              if config.num_static_real_features > 0
              else None,
              past_time_features=batch["past_time_features"].to(device),
              past_values=batch["past_values"].to(device),
              future_time_features=batch["future_time_features"].to(device),
              future_values=batch["future_values"].to(device),
              past_observed_mask=batch["past_observed_mask"].to(device),
              future_observed_mask=batch["future_observed_mask"].to(device),
          )
          loss = outputs.loss

          # Backpropagation
          accelerator.backward(loss)
          optimizer.step()

          loss_history.append(loss.item())
          if idx % 100 == 0:
              print(loss.item())

  norm_model = model

  ### Model save and Load
  model_path = f'{path}/norm_model.pth'
  torch.save(norm_model.state_dict(), model_path)

  # Action forecasting

  ## CNN + LSTM

  ### Dataset

  #### Separate train and target

  feature_col = []
  target_col = []

  for col in actions_lag_df.columns:
    if ("-" in col):
      feature_col.append(col)
    if ("+" in col):
      target_col.append(col)

  X = actions_lag_df.iloc[lag:len(actions_lag_df)-(forward - 1)][feature_col].values
  y = actions_lag_df.iloc[lag:len(actions_lag_df)-(forward - 1)][target_col].values

  X = torch.tensor(X, dtype=torch.float32)
  y = torch.tensor(y, dtype=torch.float32)

  #### With sequence dimension
  X = torch.reshape(X,(X.shape[0],lag,int(X.shape[1]/lag)))
  y = torch.reshape(y,(y.shape[0],forward,int(y.shape[1]/forward)))
  y = y.transpose(-2,-1)

  #### Bias reduction
  outliers, keep = outliers_idx(y)
  cond = to_remove_bias(y, outliers, keep)
  X = X[cond != 0]
  y = y[cond != 0]

  #### Dataloader
  X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.7, shuffle=False)
  batch_size=16
  train_loader = DataLoader(list(zip(X_train, y_train)), shuffle=True, batch_size=batch_size)
  val_loader = DataLoader(list(zip(X_val, y_val)), shuffle=False, batch_size=batch_size)

  ### Model
  hidden_dim = 100
  input_dim = X_train.shape[-2]
  nb_feature = X_train.shape[-1]
  output_dim = y_train.shape[-2]# *  y_train.shape[-1]
  num_layers = 2
  drop_prob=0.1

  model = ConvLSTMNet(input_dim=input_dim, nb_feature=nb_feature, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=output_dim, forward_steps=forward, drop_prob=drop_prob)
  model.to(device)

  lr=0.001
  nb_epochs = 5
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)

  ### Train
  # Train the model and log the progress to TensorBoard
  history = []
  model.train()

  for epoch in range(nb_epochs):
    with tqdm(train_loader, unit="batch") as tepoch:
      running_loss = 0.0
      for i, (inputs, labels) in enumerate(tepoch):

        if inputs.shape[0] != 16:
          continue
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        tepoch.set_postfix(loss=loss.item())
        if i % 100 == 99:
          torch.cuda.empty_cache()
          gc.collect()
          history.append(running_loss / 100)
          running_loss = 0.0

  action_model = model

  ### Model save and Load
  model_path = f'{path}/action_model.pth'
  torch.save(action_model.state_dict(), model_path)

def test(file_name, path=os.getcwd(), match_duration=0.5):

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  ## Process initial state
  match_df = pd.read_json(f"{path}/{file_name}",
   convert_dates=True)
  
  ### Manage "no action" label
  match_df.replace('no action', 'walk', inplace=True)

  ### One hot action encoder
  progress_column = match_df["norm"].apply(len).apply(progress_array)
  match_df.insert(len(match_df.columns), 'progress', progress_column.values)
  match_df = pd.get_dummies(match_df, columns=['label', ])

  ### Action Data

  #### Action history
  actions_col = []
  for col in match_df.columns:
    if "label_" in col:
      actions_col.append(col)

  actions_df = match_df[actions_col]

  ### Norm Data
  #### One row per norm
  continous_df = pd.DataFrame(columns=match_df.columns)
  for row_ind in match_df.index:
    duration = len(match_df["norm"][row_ind])
  for step in range(duration):
    new_row = dict()
    for column_name in match_df.columns:
      if "label_" in column_name:
        new_row[column_name] = match_df[column_name][row_ind]
      else:
        new_row[column_name] = match_df[column_name][row_ind][step]
    continous_df.loc[len(continous_df)] = new_row

  nb_features = len(continous_df.columns)

  new_cols = ['label_cross', 'label_dribble', 'label_pass', 'label_rest', 'label_run', 'label_shot', 'label_tackle', 'label_walk','norm', 'progress']
  continous_df=continous_df[new_cols]

  raw_values = continous_df[["norm","progress"]].values
  scaler = MinMaxScaler(feature_range=(-1, 1))
  scaled = scaler.fit_transform(raw_values)

  continous_df_scaled = continous_df.copy()
  continous_df_scaled["norm"] = scaled[:,0]
  continous_df_scaled["progress"] = scaled[:,1]

  ### Init batches
  if len(actions_df) >= 3:
    action_batch = torch.from_numpy(actions_df.iloc[-3:].values)
  else:
    action_batch = torch.from_numpy(actions_df.values)
    while action_batch.shape[0] < 3:
      action_batch = torch.cat((action_batch[0].unsqueeze(0),action_batch),dim=0)

  action_batch = action_batch[None,:].to(device).float()

  prediction_length = 100
  lags_sequence =  [1, 2, 3]

  aged_df = continous_df_scaled.copy()
  aged_df['age'] = 1. + aged_df.index * 1e-4

  actions_col = []
  for col in match_df.columns:
    if "label_" in col:
      actions_col.append(col)

  norm_batch = dict()
  previous = prediction_length * 2 + len(lags_sequence)
  if len(aged_df) >= previous:
    norm_batch['past_time_features'] = torch.from_numpy(aged_df.iloc[-previous:][['age'] + actions_col].values).to(device).float().unsqueeze(0)
    norm_batch['past_values'] = torch.from_numpy(aged_df.iloc[-previous:][['norm','progress']].values).to(device).float().unsqueeze(0)
  else:
    init_len = len(aged_df)
    norm_batch['past_time_features'] = torch.zeros((1,previous,9)).float().to(device)
    norm_batch['past_time_features'][0,previous - init_len:] = torch.from_numpy(aged_df[['age'] + actions_col].values).to(device).float()

    norm_batch['past_values'] = torch.zeros((1,previous,2)).float().to(device)
    norm_batch['past_values'][0,previous - init_len:] = torch.from_numpy(aged_df.iloc[-previous:][['norm','progress']].values).to(device).float()

  norm_batch['past_observed_mask'] = torch.ones((1,previous,2)).float().to(device)
  norm_batch['future_time_features'] = torch.zeros((1,prediction_length,9)).float().to(device)

  #Models

  config = InformerConfig(
      # in the multivariate setting, input_size is the number of variates in the time series per time step
      input_size=2,
      # prediction length:
      prediction_length=prediction_length,
      # context length:
      context_length=prediction_length * 2,
      # lags value copied from 1 week before:
      lags_sequence=lags_sequence,
      # we'll add 5 time features ("hour_of_day", ..., and "age"):
      num_time_features=1,
      # Dynamic real features dim
      num_dynamic_real_features = len(actions_col),
      num_parallel_samples = 1,

      # informer params:
      dropout=0.1,
      encoder_layers=6,
      decoder_layers=4,
      # project input from num_of_variates*len(lags_sequence)+num_time_features to:
      d_model=64,
  )

  action_model_path = path + '/action_model.pth'
  if not os.path.exists(action_model_path):
    print("Please train and save a action_model.pth")
    return 0
  else:
    action_model =  ConvLSTMNet(input_dim=3, nb_feature=8, hidden_dim=100, num_layers=2, output_dim=8, forward_steps=1, drop_prob=0.1)
    action_model.load_state_dict(torch.load(action_model_path,map_location=torch.device(device)))
    action_model.to(device)

  norm_model_path = path + '/norm_model.pth'
  if not os.path.exists(norm_model_path):
    print("Please train and save a norm_model.pth")
    return 0
  else:
    norm_model = InformerForPrediction(config)
    norm_model.load_state_dict(torch.load(norm_model_path,map_location=torch.device(device)))
    norm_model.to(device)

  ## Simulation loop

  init_state = [action_batch, norm_batch]

  forecasts, labels = simulation(action_model=action_model, norm_model=norm_model, duration=match_duration, init_state=init_state, device=device)

  val_unscaled = scaler.inverse_transform(forecasts)

  ## Postprocessing
  labels_list = []
  for col in match_df.columns:
    if "label_" in col:
      labels_list.append(col)


  match_out = []
  label_idx = 0
  val_unscaled = scaler.inverse_transform(forecasts)

  gait = None
  prev_prog = 0.

  for pred_idx, pred in enumerate(forecasts):
    if (pred[-1] <= -1.) and (prev_prog <= -1.):
      if gait != None:
        match_out.append(gait)
      gait = dict()
      #print("new")

      label_vec = labels[label_idx]
      #print(label_vec)
      for ind_idx, ind in enumerate(label_vec):
        if ind != 0.:
          gait["label"] = labels_list[ind_idx].split('_')[-1]

      #print(gait["label"])
      label_idx += 1

      gait["norm"] = [val_unscaled[pred_idx][-2]]


    elif gait != None:
      gait["norm"].append(val_unscaled[pred_idx][-2])

    prev_prog = pred[-1]

  match_json = json.dumps(str(match_out))

  with open(f"{path}/match_json.json", "w") as outfile:
    outfile.write(match_json)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--train", action="store_true")
  parser.add_argument("--test", action="store_true")
  parser.add_argument("--match-duration", type=float)
  parser.add_argument("--file-name", type=str)
  parser.add_argument("--path", type=str)
  
  args = parser.parse_args()

  if args.train:
    train(path=args.path)

  if args.test:
    test(file_name=args.file_name, path=args.path, match_duration=args.match_duration)


if __name__ == "__main__":
    main()
