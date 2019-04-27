from nvidia.dali.plugin.pytorch import DALIClassificationIterator
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types


class HybridTrainPipe(Pipeline):
  def __init__(self, data_dir, batch_size, num_threads=4):
    super().__init__(batch_size, num_threads, device_id=0)
    self.input = ops.FileReader(
        file_root='.', file_list='train_list', random_shuffle=True)
    self.shape = 512
    self.pre_transforms = [
        ops.nvJPEGDecoder(
            device='mixed',
            device_memory_padding=211025920,
            host_memory_padding=140544512),
        ops.Resize(
            device='gpu',
            resize_x=self.shape,
            resize_y=self.shape,
            interp_type=types.INTERP_TRIANGULAR),
    ]
    self.post_transforms = [
        ops.NormalizePermute(
            device='gpu',
            height=self.shape,
            width=self.shape,
            #mean=[105.0, 72.7, 51.8],
            #std=[255 * 6.90, 255 * 4.76, 255 * 3.38]),
            mean=[0., 0., 0.],
            std=[255., 255., 255.]),
    ]
    self.coin = ops.CoinFlip()
    self.fh_op = ops.Flip(device='gpu', horizontal=1, vertical=0)
    self.fv_op = ops.Flip(device='gpu', horizontal=0, vertical=1)
    #self.twist = ops.ColorTwist(device='gpu')
    #self.rng1 = ops.Uniform(range=(-0.1, 0.1))
    #self.rng2 = ops.Uniform(range=(0.75, 1.5))
    #self.rng3 = ops.Uniform(range=(-0.15, 0.15))

  def define_graph(self):
    images, labels = self.input(name='Reader')
    for transform in self.pre_transforms:
      images = transform(images)
    if self.coin():
      images = self.fh_op(images)
    if self.coin():
      images = self.fv_op(images)
    #images = self.twist(
    #    images,
    #    saturation=self.rng2(),
    #    contrast=self.rng2(),
    #    brightness=self.rng1(),
    #    hue=self.rng3())
    for transform in self.post_transforms:
      images = transform(images)
    return images, labels


class HybridTestPipe(Pipeline):
  def __init__(self, data_dir, batch_size, num_threads=4):
    super().__init__(batch_size, num_threads, device_id=0)
    self.input = ops.FileReader(
        file_root='.', file_list='test_list', random_shuffle=True)
    self.shape = 512
    self.transforms = [
        ops.nvJPEGDecoder(device='mixed'),
        ops.Resize(
            device='gpu',
            resize_x=self.shape,
            resize_y=self.shape,
            interp_type=types.INTERP_TRIANGULAR),
        ops.NormalizePermute(
            device='gpu',
            height=self.shape,
            width=self.shape,
            #mean=[105.0, 72.7, 51.8],
            #std=[255 * 6.90, 255 * 4.76, 255 * 3.38]),
            mean=[0., 0., 0.],
            std=[255., 255., 255.]),
    ]

  def define_graph(self):
    images, labels = self.input(name='Reader')
    for transform in self.transforms:
      images = transform(images)
    return images, labels


def get_dataloaders(batch_size, _):
  def get_loader(Pipe):
    pipe = Pipe('data', batch_size, 1)
    pipe.build()
    return DALIClassificationIterator(pipe, size=pipe.epoch_size('Reader'))

  return get_loader(HybridTrainPipe), get_loader(HybridTestPipe)
