from typing import Union, Tuple
import math
import numpy as np
import mindspore
from mindspore import nn, ops
from mindspore import Tensor, Parameter
from mindspore.ops.operations.nn_ops import WKV


class L2Wrap(nn.Cell):
    def construct(self, loss, y):
        return loss

    def bprop(self, loss, y, out, dout):
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = ops.max(y, -1, True)
        gy = ops.zeros_like(y)
        gy = ops.tensor_scatter_elements(gy, ids, maxx * factor, -1)
        return (dout, gy)


class ZeroPad2d(nn.Cell):
    def __init__(self, padding: Union[int, Tuple[int, int, int, int]]):
        super().__init__()
        if isinstance(padding, int):
            self.padding = (padding, padding, padding, padding, 0, 0)
        else:
            self.padding = padding + (0, 0)
        self.pad_op = ops.PadV3()
        self.inputs0 = Tensor(self.padding)
        self.inputs1 = Tensor(0)

    def construct(self, inputs):
        # ndim = inputs.ndim
        outputs = self.pad_op(inputs, self.inputs0, ops.Cast()(self.inputs1, inputs.dtype))
        return outputs


class RWKV_TimeMix(nn.Cell):
    def __init__(self, config, layer_id):
        super().__init__()
        dp = config.parallel_config.data_parallel
        mp = config.parallel_config.model_parallel
        self.layer_id = layer_id
        self.ctx_len = config.ctx_len
        self.n_embd = config.n_embd

        attn_sz = config.n_embd

        ratio_0_to_1 = (layer_id / (config.n_layer - 1))  # 0 to 1
        ratio_1_to_almost0 = (1.0 - (layer_id / config.n_layer))  # 1 to ~0

        # fancy time_decay
        decay_speed = np.ones(attn_sz)
        for h in range(attn_sz):
            decay_speed[h] = -5 + 8 * (h / (attn_sz - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
        self.time_decay = Parameter(Tensor(decay_speed, mindspore.float32), 'time_decay')
        # fancy time_first
        zigzag = (np.array([(i + 1) % 3 - 1 for i in range(attn_sz)]) * 0.5)
        self.time_first = Parameter(Tensor(np.ones(attn_sz) * math.log(0.3) + zigzag, mindspore.float32), 'time_first')

        # fancy time_mix
        x = np.ones((1, 1, config.n_embd))
        for i in range(config.n_embd):
            x[0, 0, i] = i / config.n_embd
        self.time_mix_k = Parameter(Tensor(np.power(x, ratio_1_to_almost0), mindspore.float32), 'time_mix_k')
        self.time_mix_v = Parameter(Tensor(np.power(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1, mindspore.float32),
                                    'time_mix_v')
        self.time_mix_r = Parameter(Tensor(np.power(x, 0.5 * ratio_1_to_almost0), mindspore.float32), 'time_mix_r')

        self.time_shift = ZeroPad2d((0, 0, 1, -1)).to_float(mindspore.float16)

        self.key = nn.Dense(config.n_embd, attn_sz, has_bias=False).to_float(mindspore.float16)
        self.key.matmul.shard(((dp, 1), (mp, 1)))
        self.value = nn.Dense(config.n_embd, attn_sz, has_bias=False).to_float(mindspore.float16)
        self.value.matmul.shard(((dp, 1), (mp, 1)))
        self.receptance = nn.Dense(config.n_embd, attn_sz, has_bias=False).to_float(mindspore.float16)
        self.receptance.matmul.shard(((dp, 1), (mp, 1)))

        self.output = nn.Dense(attn_sz, config.n_embd, has_bias=False).to_float(mindspore.float16)
        self.output.matmul.shard(((dp, mp), (1, mp)))

        self.wkv = WKV()
        self.wkv.shard(((mp,), (mp,), (dp, 1, mp), (dp, 1, mp), (dp, mp), (dp, mp), (dp, mp)))
        self.exp = ops.Exp()
        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

        self.mul = ops.Mul().shard(((dp, 1, mp), (dp, 1, mp)))
        self.sigmoid = ops.Sigmoid().shard(((dp, 1, mp),))

    def construct(self, x):
        # Mix x with the previous timestep to produce xk, xv, xr
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        # Use xk, xv, xr to produce k, v, r
        k = self.key(xk).astype(mindspore.float32)
        v = self.value(xv).astype(mindspore.float32)
        r = self.receptance(xr).astype(mindspore.float32)
        sr = self.sigmoid(r)

        bs = k.shape[0]
        a = ops.zeros([bs, self.n_embd], k.dtype)
        b = ops.zeros([bs, self.n_embd], k.dtype)
        p = ops.full([bs, self.n_embd], -1e38).astype(k.dtype)
        w = -ops.exp(self.time_decay)
        rwkv, a, b, p = self.wkv(w, self.time_first, k, v, p, a, b)
        rwkv = rwkv.astype(mindspore.float16)
        rwkv = self.mul(sr, rwkv)
        rwkv = self.output(rwkv)
        return rwkv, (a, b, p)


class RWKV_ChannelMix(nn.Cell):
    def __init__(self, config, layer_id):
        super().__init__()
        dp = config.parallel_config.data_parallel
        mp = config.parallel_config.model_parallel
        self.layer_id = layer_id

        self.time_shift = ZeroPad2d((0, 0, 1, -1)).to_float(mindspore.float16)

        ratio_1_to_almost0 = (1.0 - (layer_id / config.n_layer))  # 1 to ~0

        x = np.ones((1, 1, config.n_embd))
        for i in range(config.n_embd):
            x[0, 0, i] = i / config.n_embd

        self.time_mix_k = Parameter(Tensor(np.power(x, ratio_1_to_almost0), mindspore.float32), 'time_mix_k')
        self.time_mix_r = Parameter(Tensor(np.power(x, ratio_1_to_almost0), mindspore.float32), 'time_mix_r')
        # self.time_mix_k = Parameter(Tensor(np.power(x, ratio_1_to_almost0), mindspore.float16), 'time_mix_k')
        # self.time_mix_r = Parameter(Tensor(np.power(x, ratio_1_to_almost0), mindspore.float16), 'time_mix_r')

        hidden_sz = 4 * config.n_embd
        self.key = nn.Dense(config.n_embd, hidden_sz, has_bias=False).to_float(mindspore.float16)
        self.key.matmul.shard(((dp, 1), (mp, 1)))
        self.receptance = nn.Dense(config.n_embd, config.n_embd, has_bias=False).to_float(mindspore.float16)
        self.receptance.matmul.shard(((dp, mp), (1, mp)))
        self.value = nn.Dense(hidden_sz, config.n_embd, has_bias=False).to_float(mindspore.float16)
        self.value.matmul.shard(((dp, mp), (1, mp)))

        self.value.scale_init = 0
        self.receptance.scale_init = 0

    def construct(self, x):
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        k = self.key(xk)
        k = ops.square(ops.relu(k))
        kv = self.value(k)

        rkv = ops.sigmoid(self.receptance(xr).astype(mindspore.float32)) * kv
        return rkv


class Block(nn.Cell):
    def __init__(self, config, layer_id):
        super().__init__()
        dp = config.parallel_config.data_parallel
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm([config.n_embd], epsilon=1e-05)
        self.ln2 = nn.LayerNorm([config.n_embd], epsilon=1e-05)
        self.ln2.layer_norm.shard(((dp, 1, 1), (1,), (1,)))

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm([config.n_embd], epsilon=1e-05)

        if self.layer_id == 0 and config.model_type == 'RWKV-ffnPre':
            self.ffnPre = RWKV_ChannelMix(config, 0)
        else:
            self.att = RWKV_TimeMix(config, layer_id)

        self.ffn = RWKV_ChannelMix(config, layer_id)
        self.model_type = config.model_type

    def construct(self, x):
        state = None
        if self.layer_id == 0:
            x = self.ln0(x)
        if self.layer_id == 0 and self.model_type == 'RWKV-ffnPre':
            x = x + self.ffnPre(self.ln1(x))  # better in some cases
        else:
            x_temp, state = self.att(self.ln1(x))
            x = x + x_temp
        x = x + self.ffn(self.ln2(x))
        return x, state

