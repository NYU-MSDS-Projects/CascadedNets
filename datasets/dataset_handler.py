"""Dataset Handler."""
import os
import torch
from datasets import cifar_handler
from datasets import tinyimagenet_handler
from datasets import imagenet2012_handler
from datasets import stl10_handler


class DataHandler:
  """Handler for datasets."""

  def __init__(
      self,
      dataset_name,
      data_root,
      experiment_root,
      grayscale,
      gauss_noise,
      gauss_noise_std,
      blur,
      blur_std,
      val_split=0.1,
      test_split=0.1,
      split_idxs_root="split_idxs",
      noise_type=None,
      load_previous_splits=True,
      verbose=True, 
      **kwargs
    ):
    """Initialize dataset handler."""
    self.dataset_name = dataset_name
    self.data_root = data_root
    self.experiment_root = experiment_root
    self.test_split = test_split
    self.val_split = val_split
    self.noise_type = noise_type
    self._verbose = verbose
    self.load_previous_splits = load_previous_splits
    self.grayscale = grayscale #pg_grayscale
    self.gauss_noise = gauss_noise
    self.gauss_noise_std = gauss_noise_std
    self.blur = blur
    self.blur_std = blur_std
    self._kwargs = kwargs
    
    self._set_num_classes(dataset_name)
    
    # Set idx with dataset_name
    split_idxs_root = os.path.join(split_idxs_root, dataset_name)
    if not os.path.exists(split_idxs_root):
      os.makedirs(split_idxs_root)

    if split_idxs_root and val_split:
      self.split_idxs_root = self._build_split_idx_root(
        split_idxs_root,
        dataset_name
      )
    else:
      self.split_idxs_root = None
    
    # Create datasets
    self.datasets = self._build_datasets()

  def _set_num_classes(self, dataset_name):
    """Set number of classes in dataset."""
    if dataset_name == "CIFAR10":
      self.num_classes = 10
    elif dataset_name == "CIFAR100":
      self.num_classes = 100
    elif dataset_name == "TinyImageNet":
      self.num_classes = 200
    elif dataset_name == "STL10":
      self.num_classes = 10

  def get_transform(self, dataset_key=None):
    """Build dataset transform."""
    if dataset_key is None:
      dataset_key = list(self.datasets.keys())[0]

    normalize_transform = None
    # Grab transforms - location varies depending on base dataset.
    try:
      transforms = self.datasets[dataset_key].transform.transforms
      found = True
    except AttributeError:
      found = False

    if not found:
      try:
        transforms = self.datasets[dataset_key].dataset.transform.transforms
        found = True
      except AttributeError:
        found = False

    if not found:
      print("Transform list not found!")
    else:
      found = False
      for xform in transforms:
        if "normalize" in str(xform).lower():
          normalize_transform = xform
          found = True
          break

    if not found:
      print("Normalization transform not found!")
    return normalize_transform

  def _build_split_idx_root(self, split_idxs_root, dataset_name):
    """Build directory for split idxs."""
    if ".json" in split_idxs_root and not os.path.exists(split_idxs_root):
      split_idxs_root = os.path.join(split_idxs_root, dataset_name)
    print(f"Setting split idxs root to {split_idxs_root}")
    if not os.path.exists(split_idxs_root):
      print(f"{split_idxs_root} does not exist!")
      os.makedirs(split_idxs_root)
      print("Complete.")
    return split_idxs_root

  def _build_datasets(self):
    """Build dataset."""
    print("_build_datasets")
    print("self.grayscale", self.grayscale)
    print("self.gauss_noise", self.gauss_noise)
    print("self.gauss_noise_std", self.gauss_noise_std)
    print("self.blur", self.blur)
    print("self.blur_std", self.gauss_noise_std)
    if "cifar" in self.dataset_name.lower():
      dataset_dict = cifar_handler.create_datasets(
        self.data_root,
        dataset_name=self.dataset_name,
        val_split=self.val_split,
        grayscale=self.grayscale,
        gauss_noise=self.gauss_noise,
        gauss_noise_std=self.gauss_noise_std,
        blur=self.blur,
        blur_std=self.blur_std,
        split_idxs_root=self.split_idxs_root,
        noise_type=self.noise_type,
        load_previous_splits=self.load_previous_splits,
        verbose=self._verbose
      )

    elif self.dataset_name.lower() == "tinyimagenet":
      dataset_dict = tinyimagenet_handler.create_datasets(
        self.data_root,
        self.val_split,
        self.split_idxs_root
      )
    
    elif "stl" in self.dataset_name.lower():
        dataset_dict = stl10_handler.create_datasets(
          self.data_root,
          dataset_name=self.dataset_name,
          val_split=self.val_split,
          grayscale=self.grayscale,
          gauss_noise=self.gauss_noise,
          gauss_noise_std=self.gauss_noise_std,
          blur=self.blur,
          blur_std=self.blur_std,
          split_idxs_root=self.split_idxs_root,
          noise_type=self.noise_type,
          load_previous_splits=self.load_previous_splits,
          verbose=self._verbose
        )

    elif str.find(self.dataset_name.lower(), "imagenet2012")>-1:
      # Build path dataframe
      print("EXPERIMENT_ROOT", self.experiment_root)
      path_df = imagenet2012_handler.build_path_df(self.data_root, self.experiment_root)
      
      assert len(path_df), "Failed to load path df"
      
      print(f"size of path_df: {len(path_df)}")
      # Subset data
      if "imagenet_params" in self._kwargs:
        path_df = imagenet2012_handler.subset_path_df(
          path_df, 
          self._kwargs["imagenet_params"]
        )
      
      # Set number of classes!
      self.num_classes = path_df.class_lbl.unique().shape[0]
      print(f"# Classes: {self.num_classes}")
      # Build dataset dict
      dataset_dict = imagenet2012_handler.create_datasets(
        path_df, 
        self.val_split, 
        self.test_split,
        self.split_idxs_root,
        self.experiment_root,
        self.grayscale,
        self.gauss_noise,
        self.gauss_noise_std,
        self.blur,
        self.blur_std
      )
    
    return dataset_dict

  def build_loader(
      self,
      dataset_key,
      flags,
      dont_shuffle_train=False
    ):
    """Build dataset loader."""
    # Get dataset source
    dataset_src = self.datasets[dataset_key]
    print("DATASET_SRC", dataset_src)
    # Specify shuffling
    if dont_shuffle_train:
      shuffle = False
    else:
      shuffle = dataset_key == "train"
    
    # Creates dataloaders, which load data in batches
    loader = torch.utils.data.DataLoader(
        dataset=dataset_src,
        batch_size=flags.batch_size,
        shuffle=shuffle,
        num_workers=flags.num_workers,
        drop_last=flags.drop_last,
        pin_memory=True)
    
    return loader
