# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.utils.accelerate_utils import apply_forward_hook
from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.models.modeling_utils import ModelMixin

from .vector_quantizer import ResidualVQ
from .vector_quantizer import VectorQuantiser, VectorQuantizer

class Encoder(nn.Module):
    """A class that encapsulates the encoder."""
    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 128,
        hidden_dim: List[int] = [1024, 1024],
        dropout: float = 0.5,
        input_dropout: float = 0.4,
        residual: bool = False,
    ):
        """Constructor.

        Parameters
        ----------
        n_genes: int
            The number of genes in the gene space, representing the input dimensions.
        latent_dim: int, default: 128
            The latent space dimensions
        hidden_dim: List[int], default: [1024, 1024]
            A list of hidden layer dimensions, describing the number of layers and their dimensions.
            Hidden layers are constructed in the order of the list for the encoder and in reverse
            for the decoder.
        dropout: float, default: 0.5
            The dropout rate for hidden layers
        input_dropout: float, default: 0.4
            The dropout rate for the input layer
        residual: bool, default: False
            Use residual connections.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.network = nn.ModuleList()
        self.residual = residual
        if self.residual:
            assert len(set(hidden_dim)) == 1
        for i in range(len(hidden_dim)):
            if i == 0:  # input layer
                self.network.append(
                    nn.Sequential(
                        nn.Dropout(p=input_dropout),
                        nn.Linear(n_genes, hidden_dim[i]),
                        # nn.BatchNorm1d(hidden_dim[i]),
                        nn.LayerNorm(hidden_dim[i]),
                        nn.PReLU(),
                    )
                )
            else:  # hidden layers
                self.network.append(
                    nn.Sequential(
                        nn.Dropout(p=dropout),
                        nn.Linear(hidden_dim[i - 1], hidden_dim[i]),
                        # nn.BatchNorm1d(hidden_dim[i]),
                        nn.LayerNorm(hidden_dim[i]),
                        nn.PReLU(),
                    )
                )
        # output layer
        self.network.append(nn.Linear(hidden_dim[-1], latent_dim))
        self.layer_norm = nn.LayerNorm(latent_dim, eps=1e-12)

    def forward(self, x) -> F.Tensor:
        for i, layer in enumerate(self.network):
            if self.residual and (0 < i < len(self.network) - 1):
                x = layer(x) + x
            else:
                x = layer(x)
        # return F.normalize(x, p=2, dim=1)
        return self.layer_norm(x)
        # return x

    def save_state(self, filename: str):
        """Save state dictionary.

        Parameters
        ----------
        filename: str
            Filename to save the state dictionary.
        """
        torch.save({"state_dict": self.state_dict()}, filename)

    def load_state(self, filename: str, use_gpu: bool = False):
        """Load model state.

        Parameters
        ----------
        filename: str
            Filename containing the model state.
        use_gpu: bool
            Boolean indicating whether or not to use GPUs.
        """
        if not use_gpu:
            ckpt = torch.load(filename, map_location=torch.device("cpu"))
        else:
            ckpt = torch.load(filename)
        state_dict = ckpt['state_dict']
        first_layer_key = ['network.0.1.weight',
            'network.0.1.bias',
            'network.0.2.weight',
            'network.0.2.bias',
            'network.0.2.running_mean',
            'network.0.2.running_var',
            'network.0.2.num_batches_tracked',
            'network.0.3.weight]',]
        for key in first_layer_key:
            if key in state_dict:  
                del state_dict[key]
        self.load_state_dict(state_dict, strict=False)


class Decoder(nn.Module):
    """A class that encapsulates the decoder."""

    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 128,
        hidden_dim: List[int] = [1024, 1024],
        dropout: float = 0.5,
        residual: bool = False,
    ):
        """Constructor.

        Parameters
        ----------
        n_genes: int
            The number of genes in the gene space, representing the input dimensions.
        latent_dim: int, default: 128
            The latent space dimensions
        hidden_dim: List[int], default: [1024, 1024]
            A list of hidden layer dimensions, describing the number of layers and their dimensions.
            Hidden layers are constructed in the order of the list for the encoder and in reverse
            for the decoder.
        dropout: float, default: 0.5
            The dropout rate for hidden layers
        residual: bool, default: False
            Use residual connections.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.network = nn.ModuleList()
        self.residual = residual
        if self.residual:
            assert len(set(hidden_dim)) == 1
        for i in range(len(hidden_dim)):
            if i == 0:  # first hidden layer
                self.network.append(
                    nn.Sequential(
                        nn.Linear(latent_dim, hidden_dim[i]),
                        # nn.BatchNorm1d(hidden_dim[i]),
                        nn.LayerNorm(hidden_dim[i]),
                        nn.PReLU(),
                    )
                )
            else:  # other hidden layers
                self.network.append(
                    nn.Sequential(
                        nn.Dropout(p=dropout),
                        nn.Linear(hidden_dim[i - 1], hidden_dim[i]),
                        # nn.BatchNorm1d(hidden_dim[i]),
                        nn.LayerNorm(hidden_dim[i]),
                        nn.PReLU(),
                    )
                )
        # reconstruction layer
        self.network.append(nn.Linear(hidden_dim[-1], n_genes))

    def forward(self, x):
        for i, layer in enumerate(self.network):
            if self.residual and (0 < i < len(self.network) - 1):
                x = layer(x) + x
            else:
                x = layer(x)
        return x

    def save_state(self, filename: str):
        """Save state dictionary.

        Parameters
        ----------
        filename: str
            Filename to save the state dictionary.
        """
        torch.save({"state_dict": self.state_dict()}, filename)

    def load_state(self, filename: str, use_gpu: bool = False):
        """Load model state.

        Parameters
        ----------
        filename: str
            Filename containing the model state.
        use_gpu: bool
            Boolean indicating whether to use GPUs.
        """
        if not use_gpu:
            ckpt = torch.load(filename, map_location=torch.device("cpu"))
        else:
            ckpt = torch.load(filename)
        state_dict = ckpt['state_dict']
        last_layer_key = ['network.3.weight',
                'network.3.bias',]
        for key in last_layer_key:
            if key in state_dict:  
                del state_dict[key]
        self.load_state_dict(state_dict, strict=False)
        # self.load_state_dict(ckpt["state_dict"])

@dataclass
class VQEncoderOutput(BaseOutput):
    """
    Output of VQModel encoding method.

    Args:
        latents (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`):
            The encoded output sample from the last layer of the model.
    """

    latents: torch.Tensor


class VQModel(ModelMixin, ConfigMixin):
    r"""
    A VQ-VAE model for decoding latent representations.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (int,  *optional*, defaults to 3): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            Tuple of downsample block types.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(64,)`):
            Tuple of block output channels.
        layers_per_block (`int`, *optional*, defaults to `1`): Number of layers per block.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        latent_channels (`int`, *optional*, defaults to `3`): Number of channels in the latent space.
        sample_size (`int`, *optional*, defaults to `32`): Sample input size.
        num_vq_embeddings (`int`, *optional*, defaults to `256`): Number of codebook vectors in the VQ-VAE.
        norm_num_groups (`int`, *optional*, defaults to `32`): Number of groups for normalization layers.
        vq_embed_dim (`int`, *optional*): Hidden dim of codebook vectors in the VQ-VAE.
        scaling_factor (`float`, *optional*, defaults to `0.18215`):
            The component-wise standard deviation of the trained latent space computed using the first batch of the
            training set. This is used to scale the latent space to have unit variance when training the diffusion
            model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
            diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
            / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
            Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) paper.
        norm_type (`str`, *optional*, defaults to `"group"`):
            Type of normalization layer to use. Can be one of `"group"` or `"spatial"`.
    """

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str, ...] = ("DownEncoderBlock2D",),
        up_block_types: Tuple[str, ...] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 1,
        sample_size: int = 32,
        num_vq_embeddings: int = 256,
        norm_num_groups: int = 32,
        vq_embed_dim: Optional[int] = None,
        scaling_factor: float = 1,
        norm_type: str = "group",  # group, spatial
        mid_block_add_attention=True,
        lookup_from_codebook=False,
        force_upcast=False,
        hidden_dim = [1024,1024,1024],
        latent_dim = 128,
        num_genes = 42117,
        use_cvq = True,
        use_rvq = False,
        cvq_distance = 'cos',
        cvq_anchor='probrandom',
        quant_layers = 4,
        shared_codebook = False,
    ):
        super().__init__()

        # set cell vq vae
        self.hidden_dim = hidden_dim
        self.dropout = 0.0
        self.input_dropout = 0.0
        self.residual = False
        self.encoder = Encoder(
            n_genes=num_genes,
            latent_dim=latent_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            input_dropout=self.input_dropout,
            residual=self.residual,
        )
        self.decoder = Decoder(
            n_genes=num_genes,
            latent_dim=latent_dim,
            hidden_dim=list(reversed(self.hidden_dim)),
            dropout=self.dropout,
            residual=self.residual,
        )

        vq_embed_dim = vq_embed_dim if vq_embed_dim is not None else latent_channels
        self.vq_embed_dim = vq_embed_dim
        self.num_code = int(latent_dim/vq_embed_dim)
        self.latent_dim = latent_dim

        self.theta = torch.nn.Parameter(torch.randn(num_genes), requires_grad=True)

        if use_cvq:
            self.quantize = VectorQuantiser(num_vq_embeddings, vq_embed_dim, beta=0.25, distance=cvq_distance, anchor=cvq_anchor, first_batch=False, contras_loss=True)
        elif use_rvq:
            self.quantize = ResidualVQ(
                dim = vq_embed_dim,
                num_quantizers = quant_layers,
                codebook_size = num_vq_embeddings,
                # learnable_codebook = True,
                # ema_update = True,
                # stochastic_sample_codes = True,
                # sample_codebook_temp = 0.1,         # temperature for stochastically sampling codes, 0 would be equivalent to non-stochastic
                shared_codebook = shared_codebook,              # whether to share the codebooks for all quantizers or not
                commitment_weight=1,                    # 1 1
                orthogonal_reg_weight=1e-3,             # 开启 orthogonal reg loss 1e-3 1e-2
                orthogonal_reg_active_codes_only=True,  # 只对活跃码字做正交
                codebook_diversity_loss_weight=1e-3,    # 开启 diversity loss 1e-3 1e-2
                codebook_diversity_temperature=100.,
                in_place_codebook_optimizer=lambda params: torch.optim.AdamW(params, lr=1e-4) # 开启 inplace optimize
            )
        else:
            self.quantize = VectorQuantizer(num_vq_embeddings, vq_embed_dim, beta=0.25, remap=None, sane_index_shape=False)

        self.relu = nn.ReLU()


    @apply_forward_hook
    def encode(self, x: torch.Tensor, return_dict: bool = True, vq: bool = True, gene_id: torch.Tensor = None) -> VQEncoderOutput:
        h = self.encoder(x)
        if vq:
            h = h.view(-1, self.num_code, self.vq_embed_dim)

        if not return_dict:
            return (h,)

        return VQEncoderOutput(latents=h)

    @apply_forward_hook
    def decode(
        self, h: torch.Tensor, force_not_quantize: bool = False, return_dict: bool = True, shape=None
    ) -> Union[DecoderOutput, torch.Tensor]:
        # also go through quantization layer
        if not force_not_quantize:
            quant, commit_loss, _ = self.quantize(h)
        elif self.config.lookup_from_codebook:
            quant = self.quantize.get_codebook_entry(h, shape)
            commit_loss = torch.zeros((h.shape[0])).to(h.device, dtype=h.dtype)
        else:
            quant = h
            commit_loss = torch.zeros((h.shape[0])).to(h.device, dtype=h.dtype)
        # quant2 = self.post_quant_conv(quant)
        if not force_not_quantize:
            quant = quant.reshape(-1, self.latent_dim)
        dec = self.decoder(quant)

        if not return_dict:
            return dec, commit_loss

        return DecoderOutput(sample=dec, commit_loss=commit_loss)

    def norm_total(self, array, target_sum = 1e4):        
        current_sum = array.sample.sum(axis=1)
        normalization_factor = target_sum / current_sum  
        array.sample = array.sample * normalization_factor.unsqueeze(-1)  
        return array
    
    def forward(
        self, sample: torch.Tensor, return_dict: bool = True, mode: str = 'rna', vq: bool = True, gene_id: torch.Tensor = None
    ) -> Union[DecoderOutput, Tuple[torch.Tensor, ...]]:
        r"""
        The [`VQModel`] forward method.

        Args:
            sample (`torch.Tensor`): Input sample.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.autoencoders.vq_model.VQEncoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.autoencoders.vq_model.VQEncoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.autoencoders.vq_model.VQEncoderOutput`] is returned, otherwise a
                plain `tuple` is returned.
        """

        h = self.encode(sample,vq=vq, gene_id=gene_id).latents
        dec = self.decode(h,force_not_quantize = not vq)

        if mode == 'atac':
            dec.sample =  F.sigmoid(dec.sample)

        if not return_dict:
            return dec.sample, dec.commit_loss
        return dec
