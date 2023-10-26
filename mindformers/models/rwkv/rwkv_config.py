# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Rwkv Config API."""

from mindformers.modules.transformer.moe import MoEConfig
from mindformers.modules.transformer.transformer import default_transformer_config, default_moe_config, \
    TransformerOpParallelConfig
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from ..utils import convert_mstype
from ..base_config import BaseConfig
from ...mindformer_book import MindFormerBook

__all__ = ['RWKVConfig']


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class RWKVConfig(BaseConfig):
    """
    rwkv config class which defines the model size

    """

    _support_list = MindFormerBook.get_config_support_list()['rwkv']

    def __init__(self,
                 ctx_len: int = 1024,
                 vocab_size: int = 50277,
                 n_embd: int = 768,
                 n_layer: int = 12,
                 init_lr: str = '',
                 parallel_config: TransformerOpParallelConfig = default_transformer_config,
                 checkpoint_name_or_path: str = '',
                 moe_config: MoEConfig = default_moe_config,
                 **kwargs):
        super(RWKVConfig, self).__init__(**kwargs)
        self.ctx_len = ctx_len
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.init_lr = init_lr
        self.parallel_config = parallel_config
        self.checkpoint_name_or_path = checkpoint_name_or_path
        self.moe_config = moe_config
