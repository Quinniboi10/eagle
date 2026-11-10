# Eagle
> Eagle is a PyTorch abstraction layer offering simplicity without detracting from capability

## Getting Started
While running the Python installation or environment you wish to use, execute the below commands
```bash
git clone https://github.com/Quinniboi10/eagle.git
cd eagle
pip install -e .
```

## Basic usage
A simple demonstration of Eagle's capability
```py
import torch
from torch import nn
from eagle import Learner

# Define your model, optimizer, and train/valid loader here

learner = Learner(model, optimizer, train_loader, valid_loader, loss_fn=nn.CrossEntropyLoss())
learner.fit_one_cycle(10)  # 10 epochs
learner.save("./exports/model")
learner.recorder.plot("train loss", "valid loss")
```

## Inside `Learner`
The `Learner` type exposes tons of internal values for better transparency when using `Callback`s and `Metric`s.
```py
Learner
  self.model:              Model
  self.optim:              Optimizer
  self.dl:                 Train dataloader
  self.valid_dl:           Validation dataloader
  self.loss_fn:            Loss function
  self.clip_grads:         Value to which gradients will be clipped
  self.accum:              Number of sub-minibatches to accumulate before applying updates
  self.autocast:           Boolean to enable/disable PyTorch's automatic type casting
  self.schedule:           PyTorch LRScheduler used in training (ignored while using fit_one_cycle)

  # Values below should be treated as read-only
  self.num_train_batches:  Training batches per epoch
  self.num_valid_batches:  Validation batches per epoch
  self.train_bs:           Train batch size
  self.valid_bs:           Validation batch size
  self.cbs:                Eagle callbacks
  self.device:             Automatically resolved target device (can override with Learner.to)
  self.state:              Eagle.Metric.State.TRAIN/VALID
  self.x:                  Input tensor
  self.y:                  Target tensor
  self.preds:              Output/prediction tensor
  self.lr:                 Last LR used in training
  self.train_loss:         Last training loss value
  self.epoch:              Current epoch
  self.valid_loss:         Last valid loss value

  # Functions are listed here, for up-to-date arguments, refer to the functions themselves. PyLance should handle this for you!
  __init__(model, optimizer, train_dl, valid_dl, loss_fn, metrics=None, cbs=None, clip_grads=None, accum=1, autocast=True)
  """
    model: the PyTorch model to train
    optimizer: the PyTorch optimizer to run
    train_dl: the PyTorch train dataloader
    valid_dl: the PyTorch valid dataloader
    loss_fn: An extension of nn.Module that implements forward(preds, targets)
    metrics: Which metrics to use apart from LR, train loss, valid loss
    cbs: Which callbacks to attach after Recorder, TrainEval, State; defaults to Status (None)
    clip_grads: Clip gradients before optimizer.step(); defaults to off (None)
    accum: How many batches to average loss across before running optimizer.step(). Simulates true_bs = train_bs * accum
    autocast: True/False for using PyTorch's built-in automatic type casting to save memory
  """
  to(device)
  """
    device: PyTorch device (as str) to move model, data, and training to
  """
  fit_one_cycle(num_epochs, max_lr=1e-3, div=25, final_div=1e5)
  """
    num_epochs: N epochs to train
    max_lr, div, final_div: refer to PyTorch's documentation for torch.optim.lr_scheduler.OneCycleLR
  """
  fit(num_epochs, use_internal_schedule=False)
  """
    num_epochs: N epochs to train
    use_internal_schedule: If True, uses self.schedule, otherwise uses the optimizer's constant LR
  """
  save(path, silent=False)
  """
    Saves the model's state dict using torch.save(self.model.state_dict())

    path: Path or str to which the file should be saved
    silent: Prints a line indicating the model was saved if False
  """
  export(path, *args)
  """
    Uses torch.export.export(self.model, *args). If no args are given, just uses the last train/valid batch to calculate a graph.
    NOTE: Highly recommended to pass your own inputs. Using the last given graph can cause issues with batch dimensions.
    
    path: Path or str to which the file should be saved
    *args: Arguments passed to torch.export.export (as shown above)
  """
  add_cb(cb)
  """
    Registers a new callback, raises RuntimeError the name already exists

    cb: Callback to register
  """
  param_info(sort_by="count")
  """
    Pretty-prints info about what contributes to the model's parameters

    sort_by: "order" or "count", either prints in order of use in a feed-forward or by # parameters
  """
```

## What are `Callback`s?
Callbacks are small pieces of code which can be executed at any of the positions listed below
```py
BEFORE_FIT
BEFORE_TRAIN
BEFORE_VALID
BEFORE_BATCH

AFTER_FIT
AFTER_TRAIN
AFTER_VALID
AFTER_BATCH
```

You can add your own callbacks by calling `learner.add_cb(MyCallback())` or passing `cbs=[MyCallback()]` to the constructor. Some callbacks (`RecorderCallback, TrainEvalCallback, StateCallback`) will always be added.

### Implementing a Callback
Callbacks are intentionally made to be simple! They extend Callback and have full access to the Learner they're registered to. Below are some simple examples to get you started.
```py
# Toggles train/eval mode
class TrainEvalCallback(Callback):
    def before_train(self):
        self.learner.model.train()
    def before_valid(self):
        self.learner.model.eval()

# Switches the learner.state
class StateCallback(Callback):
    def before_train(self):
        self.learner.state = State.TRAIN
    def before_valid(self):
        self.learner.state = State.VALID
```

### Notes
The most important callback is the RecorderCallback. It manages metrics attached to the Learner, and internally stores and manages a list of `Snapshot`s.

Critically, the RecorderCallback implements `plot(*metric_names, y_label=None, units: Literal["batch", "epoch"]="epoch")` to allow graphing of any metric after training completes.

## Metrics
Metrics track the internal state of the learner and are contained under the `Learner.recorder` callback. Some metrics (`LRMetric, TrainLossMetric, ValidLossMetric`) are always enabled. Further metrics can be added by passing them under `metrics=[MyMetric()]` to the constructor.

### Implementing a Metric
Metrics extend `Metric` and define a state in which their values should be captured (`State.TRAIN/VALID/BOTH`).
```py
class LRMetric(Metric):
    state = State.TRAIN
    do_reporting = False # Here we disable reporting so StatusCallback won't print it, but the value is still saved in case you'd like to plot it
    name = "lr"

    @property
    def value(self):
        return self.learner.lr
```

### EMA Metrics
Some values (like train loss) are noisy. To resolve this, extend `ExpAvgMetric` to automatically apply EMA to incoming values.
```py
class TrainLossMetric(ExpAvgMetric):
    state = State.TRAIN
    do_reporting = True
    name = "train loss"

    def step(self):
        self.avg.step(self.learner.train_loss)
```
