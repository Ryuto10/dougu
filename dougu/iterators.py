from math import ceil

import torchtext


class PaddingBucketIterator(object):
    """Generate padded mini-batches to minimize padding as much as possible.
    Args:
        dataset: The list of the instances
            ex) dataset = [(x, y), ...]
        sort_key: The function for sorting the instances
            ex) In the case of the instance (x, y)
                def sort_key(instance):
                    return len(instance[0])
        batch_size: The size of mini-batch
        shuffle: whether to shuffle instance
        padding_value: The value for padding
    """

    def __init__(self,
                 dataset=None,
                 sort_key=None,
                 batch_size: int = 128,
                 shuffle: bool = False,
                 padding_value: int = 0):
        self.dataset = dataset
        self.sort_key = sort_key
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.padding_value = padding_value

        self.iterator = None

        if self.dataset is not None:
            self.create_iterator()
            self.create_batches()

    def __len__(self):
        return ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):
        return self

    def __next__(self):
        return self.padding(next(self.iterator.batches))

    def create_iterator(self):
        self.iterator = torchtext.data.BucketIterator(self.dataset,
                                                      batch_size=self.batch_size,
                                                      sort_key=self.sort_key,
                                                      shuffle=self.shuffle,
                                                      sort_within_batch=True)
        self.create_batches()

    def create_batches(self):
        self.iterator.create_batches()

    def padding(self, batch):
        """Return a padded mini-batch
        Example:
            The example of using 'torch.nn.utils.rnn.pad_sequence':
            Args:
                batch: [[xs, ys], ...], length = batch size
            Returns:
                padded_xs: torch.Tensor, shape = (batch, seq_length)
                ys: [y, ...], length = batch size

            xs, ys = zip(*batch)
            padded_xs = pad_sequence(xs, batch_first=True, padding_value=self.padding_value)

            return [padded_xs, ys]
        """
        raise NotImplementedError
