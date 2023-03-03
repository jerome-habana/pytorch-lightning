from unittest.mock import Mock

import pytest
import torch
from tests_fabric.helpers.models import BoringFabric
from tests_fabric.helpers.runif import RunIf

from lightning.fabric.strategies.parallel_hpu import HPUParallelStrategy
from lightning.fabric.strategies.single_hpu import SingleHPUStrategy
from lightning.fabric.wrappers import _FabricModule, _FabricOptimizer

@RunIf(hpu=True)
def test_single_device_default_device():
    assert SingleHPUStrategy().root_device == torch.device("hpu")

@RunIf(hpu=True)
@pytest.mark.parametrize(
    ["process_group_backend", "device_str", "expected_process_group_backend"],
    [
        pytest.param(None, "hpu:0", "hccl"),
        #pytest.param(None, "cpu", "gloo"),
    ],
)
def test_hpu_parallel_process_group_backend(process_group_backend, device_str, expected_process_group_backend):
    """Test settings for process group backend."""

    class MockHPUParallelStrategyStrategy(HPUParallelStrategy):
        def __init__(self, root_device, process_group_backend):
            self._root_device = root_device
            super().__init__(process_group_backend=process_group_backend)

        @property
        def root_device(self):
            return self._root_device

    strategy = MockHPUParallelStrategyStrategy(process_group_backend=process_group_backend, root_device=torch.device(device_str))
    assert strategy._get_process_group_backend() == expected_process_group_backend