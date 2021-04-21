import numbers
from typing import Any, Callable, Optional, Union

import torch

from ignite.metrics import Metric, VariableAccumulation
from ignite.exceptions import NotComputableError


# Same as original Ignite implementation, except no @sync_all_reduce.
# We remove this since we're calculating metrics in the main process only,
# which leads to @sync_all_reduce to hang because the processes it is
# waiting for do not exist
class Average(VariableAccumulation):
    """Helper class to compute arithmetic average of a single variable.
    - ``update`` must receive output of the form `x`.
    - `x` can be a number or `torch.Tensor`.
    Note:
        Number of samples is updated following the rule:
        - `+1` if input is a number
        - `+1` if input is a 1D `torch.Tensor`
        - `+batch_size` if input is an ND `torch.Tensor`. Batch size is the first dimension (`shape[0]`).
        For input `x` being an ND `torch.Tensor` with N > 1, the first dimension is seen as the number of samples and
        is summed up and added to the accumulator: `accumulator += x.sum(dim=0)`
    Examples:
    .. code-block:: python
        evaluator = ...
        custom_var_mean = Average(output_transform=lambda output: output['custom_var'])
        custom_var_mean.attach(evaluator, 'mean_custom_var')
        state = evaluator.run(dataset)
        # state.metrics['mean_custom_var'] -> average of output['custom_var']
    Args:
        output_transform (callable, optional): a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
        device (str of torch.device, optional): optional device specification for internal storage.
    """

    def __init__(self, output_transform: Callable = lambda x: x, device: Optional[Union[str, torch.device]] = 'cpu'):
        def _mean_op(a, x):
            if isinstance(x, torch.Tensor) and x.ndim > 1:
                x = x.sum(dim=0)
            return a + x

        super(Average, self).__init__(op=_mean_op, output_transform=output_transform, device=device)

    def compute(self) -> Union[Any, torch.Tensor, numbers.Number]:
        if self.num_examples < 1:
            raise NotComputableError(
                "{} must have at least one example before" " it can be computed.".format(self.__class__.__name__)
            )

        return self.accumulator / self.num_examples


class TopKCategoricalAccuracy(Metric):
    """
    Calculates the top-k categorical accuracy.

    - `update` must receive output of the form `(y_pred, y)`.
    """
    def __init__(
            self, k=5, output_transform=lambda x: x, already_sorted=False):
        super(TopKCategoricalAccuracy, self).__init__(output_transform)
        self._k = k
        self.already_sorted = already_sorted

    def reset(self):
        self._num_correct = 0
        self._num_examples = 0

    def update(self, output):
        y_pred, y = output
        if self.already_sorted:
            sorted_indices = y_pred[:, :self._k]
        else:
            sorted_indices = torch.topk(y_pred, self._k, dim=1)[1]
        expanded_y = y.view(-1, 1).expand(-1, self._k)
        correct = torch.sum(torch.eq(sorted_indices, expanded_y), dim=1)
        self._num_correct += torch.sum(correct).item()
        self._num_examples += correct.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError("TopKCategoricalAccuracy must have at"
                                     "least one example before it can be computed.")
        return self._num_correct / self._num_examples
