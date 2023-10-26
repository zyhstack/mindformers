from typing import Union, Tuple
import math
import numpy as np
import mindspore
from mindspore import nn, ops
from mindspore import Tensor
from mindspore.common.initializer import initializer, Zero, Uniform, Orthogonal

from mindformers.modules.transformer.moe import default_moe_config
from mindformers.modules.layers import LayerNorm, Dropout, Linear
from mindformers.core.loss import CrossEntropyLoss
from mindformers.modules.transformer import AttentionMask, VocabEmbedding
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.models.base_model import BaseModel
from mindformers.mindformer_book import MindFormerBook
from mindformers.tools.logger import logger
from mindformers.pet import LoraAdapter, PetAdapter
from .rwkv_config import RWKVConfig
from .rwkv_modules import L2Wrap, ZeroPad2d, RWKV_TimeMix, RWKV_ChannelMix, Block

__all__ = ['RWKVModel']


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class RWKVModel(BaseModel):
    r"""
        Provide RWKVModel training loss or logits through network.
        Args:
            config (RWKVConfig): The config of RWKVModel.

        Returns:
            Tensor, the loss or logits of the network.

        """
    _support_list = MindFormerBook.get_model_support_list()['rwkv']

    def __init__(self, config: RWKVConfig = None):
        config = config if config is not None else RWKVConfig()
        super(RWKVModel, self).__init__(config, auto_prefix=True)
        dp = config.parallel_config.data_parallel
        mp = config.parallel_config.model_parallel
        self.stridedslice = ops.StridedSlice().shard(((dp, 1),))

        self.emb = nn.Embedding(config.vocab_size, config.n_embd).to_float(mindspore.float16)

        # self.blocks = nn.CellList([Block(config, i) for i in range(config.n_layer)])
        self.blocks = nn.CellList()
        for i in range(config.n_layer):
            wkv_block = Block(config, i)
            if config.recompute:
                wkv_block.recompute()
            self.blocks.append(wkv_block)
        self.ln_out = nn.LayerNorm([config.n_embd], epsilon=1e-05)
        self.head = nn.Dense(config.n_embd, config.vocab_size, has_bias=False).to_float(mindspore.float16)
        self.ctx_len = config.ctx_len
        self.vocab_size = config.vocab_size
        self.l2_wrapper = L2Wrap()
        if config.recompute:
            self.emb.recompute()
            self.ln_out.recompute()
            self.head.recompute()
        self.config = config

    def construct(self, input_ids, attention_mask=None, labels=None):
        batch_size, seq_length = input_ids.shape
        assert seq_length <= self.ctx_len + 1, "Cannot forward, because len(input) > model ctx_len."
        tokens = self.stridedslice(input_ids, (0, 0), (batch_size, seq_length - 1), (1, 1))
        labels = self.stridedslice(input_ids, (0, 1), (batch_size, seq_length), (1, 1))

        all_states = ()
        x = self.emb(tokens)

        for block in self.blocks:
            x, state = block(x)
            all_states += (state,)

        x = self.ln_out(x)
        x = self.head(x)
        x = x.astype(mindspore.float32)

        loss = ops.cross_entropy(x.view(-1, x.shape[-1]), labels.view(-1))
        return self.l2_wrapper(loss, x)

    def _generate_init_weight(self, lr_init):
        params_dict = self.parameters_dict()
        for k, value in params_dict.items():
            shape = value.shape
            dtype = value.dtype
            gain = 1.0
            scale = 1.0
            if "ln_" in k or ".ln" in k or "time_" in k or "_mask" in k or "pos_emb" in k or '.mask.' in k:
                if 'ln_x.gamma' in k:
                    layer_scale = (1 + int(k.split('.')[1])) / self.config.n_layer
                    new_param = (value * 0.0) + (layer_scale ** 0.7)
                    value.set_data(Tensor(new_param, dtype))
            else:
                if "emb.embedding_table" in k:
                    scale = -1 * float(lr_init)
                else:
                    if shape[0] > shape[1]:
                        gain = math.sqrt(shape[0] / shape[1])
                    for kk in [".att.key.", ".att.receptance.", ".att.output.", ".att.key.", ".ffn.value.",
                               ".ffn.receptance.", ".ffnPre.value.", ".ffnPre.receptance.", "head_q.", '.oo.', '.rr.']:
                        if kk in k:
                            scale = 0
                    if k == "model.head.weight":
                        scale = 0.5
                    if "head_k." in k:
                        scale = 0.1
                    if "head_q." in k:
                        scale = 0

                print(f"{str(shape[0]).ljust(5)} {str(shape[1]).ljust(5)} {str(scale).ljust(4)} {k}")

                if scale == 0:
                    value.set_data(initializer(Zero(), shape, dtype))
                elif scale < 0:
                    value.set_data(initializer(Uniform(scale), shape, dtype))
                else:
                    value.set_data(initializer(Orthogonal(gain=gain * scale), shape, dtype))
