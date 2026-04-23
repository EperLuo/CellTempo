from __future__ import annotations

import random
from math import ceil
from functools import partial, cache
from itertools import zip_longest

import torch
from torch import nn, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F
import torch.distributed as dist

from typing import Optional, Tuple, Union, List, TypeVar
import numpy as np

from einops import rearrange, repeat, reduce, pack, unpack

from einx import get_at

class VectorQuantizer(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly avoids costly matrix
    multiplications and allows for post-hoc remapping of indices.
    """

    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(
        self,
        n_e: int,
        vq_embed_dim: int,
        beta: float,
        remap=None,
        unknown_index: str = "random",
        sane_index_shape: bool = False,
        legacy: bool = True,
    ):
        super().__init__()
        self.n_e = n_e
        self.vq_embed_dim = vq_embed_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.vq_embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.used: torch.Tensor
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index  # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            print(
                f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                f"Using {self.unknown_index} for unknown indices."
            )
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds: torch.LongTensor) -> torch.LongTensor:
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds: torch.LongTensor) -> torch.LongTensor:
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:  # extra token
            inds[inds >= self.used.shape[0]] = 0  # simply set to zero
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple]:
        # reshape z -> (batch, height, width, channel) and flatten
        # z = z.permute(0, 1, 2).contiguous()
        z_flattened = z.reshape(-1, self.vq_embed_dim)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        min_encoding_indices = torch.argmin(torch.cdist(z_flattened, self.embedding.weight), dim=1)

        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q: torch.Tensor = z + (z_q - z).detach()

        # reshape back to match original input shape
        # z_q = z_q.permute(0, 2, 1).contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0], -1)  # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)  # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices: torch.LongTensor, shape: Tuple[int, ...]) -> torch.Tensor:
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0], -1)  # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1)  # flatten again

        # get quantized latent vectors
        z_q: torch.Tensor = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q

class FeaturePool():
    """
    This class implements a feature buffer that stores previously encoded features

    This buffer enables us to initialize the codebook using a history of generated features
    rather than the ones produced by the latest encoders
    """
    def __init__(self, pool_size, dim=64):
        """
        Initialize the FeaturePool class

        Parameters:
            pool_size(int) -- the size of featue buffer
        """
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.nums_features = 0
            self.features = (torch.rand((pool_size, dim)) * 2 - 1)/ pool_size

    def query(self, features):
        """
        return features from the pool
        """
        self.features = self.features.to(features.device)    
        if self.nums_features < self.pool_size:
            if features.size(0) > self.pool_size: # if the batch size is large enough, directly update the whole codebook
                random_feat_id = torch.randint(0, features.size(0), (int(self.pool_size),))
                self.features = features[random_feat_id]
                self.nums_features = self.pool_size
            else:
                # if the mini-batch is not large nuough, just store it for the next update
                num = self.nums_features + features.size(0)
                self.features[self.nums_features:num] = features
                self.nums_features = num
        else:
            if features.size(0) > int(self.pool_size):
                random_feat_id = torch.randint(0, features.size(0), (int(self.pool_size),))
                self.features = features[random_feat_id]
            else:
                random_id = torch.randperm(self.pool_size)
                self.features[random_id[:features.size(0)]] = features

        return self.features

class VectorQuantiser(nn.Module):
    """
    Improved version over vector quantiser, with the dynamic initialisation
    for these unoptimised "dead" points.
    num_embed: number of codebook entry
    embed_dim: dimensionality of codebook entry
    beta: weight for the commitment loss
    distance: distance for looking up the closest code
    anchor: anchor sampled methods
    first_batch: if true, the offline version of our model
    contras_loss: if true, use the contras_loss to further improve the performance
    """
    def __init__(self, n_e, vq_embed_dim, beta, distance='cos', 
                 anchor='probrandom', first_batch=False, contras_loss=False):
        super().__init__()

        self.num_embed = n_e
        self.embed_dim = vq_embed_dim
        self.beta = beta
        self.distance = distance
        self.anchor = anchor
        self.first_batch = first_batch
        self.contras_loss = contras_loss
        self.decay = 0.99
        self.init = False

        self.pool = FeaturePool(self.num_embed, self.embed_dim)
        self.embedding = nn.Embedding(self.num_embed, self.embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embed, 1.0 / self.num_embed)
        self.register_buffer("embed_prob", torch.zeros(self.num_embed))
    
    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        # z = rearrange(z, 'b c h w -> b h w c').contiguous()
        # z_flattened = z.view(-1, self.embed_dim)
        z_flattened = z.reshape(-1, self.embed_dim)

        # clculate the distance
        if self.distance == 'l2':
            # l2 distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
            d = - torch.sum(z_flattened.detach() ** 2, dim=1, keepdim=True) - \
                torch.sum(self.embedding.weight ** 2, dim=1) + \
                2 * torch.einsum('bd, dn-> bn', z_flattened.detach(), rearrange(self.embedding.weight, 'n d-> d n'))
        elif self.distance == 'cos':
            # cosine distances from z to embeddings e_j 
            normed_z_flattened = F.normalize(z_flattened, dim=1).detach()
            normed_codebook = F.normalize(self.embedding.weight, dim=1)
            d = torch.einsum('bd,dn->bn', normed_z_flattened, rearrange(normed_codebook, 'n d -> d n'))

        # encoding
        sort_distance, indices = d.sort(dim=1)
        # look up the closest point for the indices
        encoding_indices = indices[:,-1]
        encodings = torch.zeros(encoding_indices.unsqueeze(1).shape[0], self.num_embed, device=z.device)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)

        # quantise and unflatten
        z_q = torch.matmul(encodings.to(z.dtype), self.embedding.weight).view(z.shape)
        # compute loss for embedding
        loss = self.beta * torch.mean((z_q.detach()-z)**2) + torch.mean((z_q - z.detach()) ** 2)
        # preserve gradients
        z_q = z + (z_q - z).detach()
        # reshape back to match original input shape
        # z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        # count
        # import pdb
        # pdb.set_trace()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        min_encodings = encodings

        # online clustered reinitialisation for unoptimized points
        if self.training:
            # calculate the average usage of code entries
            self.embed_prob.mul_(self.decay).add_(avg_probs, alpha= 1 - self.decay)
            # running average updates
            if self.anchor in ['closest', 'random', 'probrandom'] and (not self.init):
                # closest sampling
                if self.anchor == 'closest':
                    sort_distance, indices = d.sort(dim=0)
                    random_feat = z_flattened.detach()[indices[-1,:]]
                # feature pool based random sampling
                elif self.anchor == 'random':
                    random_feat = self.pool.query(z_flattened.detach())
                # probabilitical based random sampling
                elif self.anchor == 'probrandom':
                    norm_distance = F.softmax(d.t(), dim=1)
                    prob = torch.multinomial(norm_distance, num_samples=1).view(-1)
                    random_feat = z_flattened.detach()[prob]
                # decay parameter based on the average usage
                decay = torch.exp(-(self.embed_prob*self.num_embed*10)/(1-self.decay)-1e-3).unsqueeze(1).repeat(1, self.embed_dim)
                self.embedding.weight.data = self.embedding.weight.data * (1 - decay) + random_feat * decay
                if self.first_batch:
                    self.init = True
            # contrastive loss
            if self.contras_loss:
                sort_distance, indices = d.sort(dim=0)
                dis_pos = sort_distance[-max(1, int(sort_distance.size(0)/self.num_embed)):,:].mean(dim=0, keepdim=True)
                dis_neg = sort_distance[:int(sort_distance.size(0)*1/2),:]
                dis = torch.cat([dis_pos, dis_neg], dim=0).t() / 0.07
                contra_loss = F.cross_entropy(dis, torch.zeros((dis.size(0),), dtype=torch.long, device=dis.device))
                loss +=  contra_loss

        return z_q, loss, (perplexity, min_encodings, encoding_indices)



# helper functions

def exists(val):
    return val is not None

def first(it):
    return it[0]

def default(val, d):
    return val if exists(val) else d

def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

def unique(arr):
    return list({*arr})

def round_up_multiple(num, mult):
    return ceil(num / mult) * mult

# distributed helpers

def is_distributed():
    return dist.is_initialized() and dist.get_world_size() > 1

def get_maybe_sync_seed(device, max_size = 10_000):
    rand_int = torch.randint(0, max_size, (), device = device)

    if is_distributed():
        dist.all_reduce(rand_int)

    return rand_int.item()

# the mlp for generating the neural implicit codebook
# from Huijben et al. https://arxiv.org/abs/2401.14732

class MLP(Module):
    def __init__(
        self,
        dim,
        dim_hidden = None,
        depth = 4,             # they used 4 layers in the paper
        l2norm_output = False
    ):
        super().__init__()
        dim_hidden = default(dim_hidden, dim)

        self.proj_in = nn.Linear(2 * dim, dim)

        layers = ModuleList([])

        for _ in range(depth):
            layers.append(nn.Sequential(
                nn.Linear(dim, dim_hidden),
                nn.SiLU(),
                nn.Linear(dim_hidden, dim)
            ))

        self.layers = layers
        self.l2norm_output = l2norm_output

    def forward(
        self,
        codes,
        *,
        condition
    ):
        one_headed = codes.ndim == 2

        condition, _ = pack([condition], 'b * d') # handle condition with ndim 2 - (batch, dim)

        if one_headed:
            codes = rearrange(codes, 'c d -> 1 c d')

        heads, num_codes, batch, seq_len = codes.shape[0], codes.shape[-2], condition.shape[0], condition.shape[-2]

        codes = repeat(codes, 'h c d -> h b n c d', n = seq_len, b = batch)
        condition = repeat(condition, 'b n d -> h b n c d', c = num_codes, h = heads)

        x = torch.cat((condition, codes), dim = -1)
        x = self.proj_in(x)

        for layer in self.layers:
            x = layer(x) + x

        if self.l2norm_output:
            x = F.normalize(x, dim = -1)

        if not one_headed:
            return x

        return rearrange(x, '1 ... -> ...')

# main class

class ResidualVQ(Module):
    """ Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf """
    def __init__(
        self,
        *,
        dim,
        num_quantizers: int | None = None,
        codebook_size: int | tuple[int, ...],
        codebook_dim = None,
        shared_codebook = False,
        heads = 1,
        quantize_dropout = False,
        quantize_dropout_cutoff_index = 0,
        quantize_dropout_multiple_of = 1,
        accept_image_fmap = False,
        implicit_neural_codebook = False, # QINCo from https://arxiv.org/abs/2401.14732
        mlp_kwargs: dict = dict(),
        **vq_kwargs
    ):
        super().__init__()
        assert heads == 1, 'residual vq is not compatible with multi-headed codes'
        assert exists(num_quantizers) or isinstance(codebook_size, tuple)

        codebook_dim = default(codebook_dim, dim)
        codebook_input_dim = codebook_dim * heads

        requires_projection = codebook_input_dim != dim
        self.project_in = nn.Linear(dim, codebook_input_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(codebook_input_dim, dim) if requires_projection else nn.Identity()
        self.has_projections = requires_projection

        self.accept_image_fmap = accept_image_fmap

        self.implicit_neural_codebook = implicit_neural_codebook

        if implicit_neural_codebook:
            vq_kwargs.update(
                learnable_codebook = True,
                ema_update = False
            )

        if shared_codebook:
            vq_kwargs.update(
                manual_ema_update = True,
                manual_in_place_optimizer_update = True
            )

        # take care of maybe different codebook sizes across depth

        codebook_sizes = cast_tuple(codebook_size, num_quantizers)

        num_quantizers = default(num_quantizers, len(codebook_sizes))
        assert len(codebook_sizes) == num_quantizers

        self.num_quantizers = num_quantizers

        self.codebook_sizes = codebook_sizes
        self.uniform_codebook_size = len(unique(codebook_sizes)) == 1

        # define vq across layers
        self.layers = ModuleList([VectorQuantiser(layer_codebook_size, codebook_dim, beta=0.25, distance='cos', anchor='probrandom', first_batch=False, contras_loss=True) for layer_codebook_size in codebook_sizes])

        # assert all([not vq.has_projections for vq in self.layers])

        self.quantize_dropout = quantize_dropout and num_quantizers > 1

        assert quantize_dropout_cutoff_index >= 0

        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index
        self.quantize_dropout_multiple_of = quantize_dropout_multiple_of  # encodec paper proposes structured dropout, believe this was set to 4

        # setting up the MLPs for implicit neural codebooks
        
        self.mlps = None

        if implicit_neural_codebook:
            self.mlps = ModuleList([MLP(dim = codebook_dim, l2norm_output = first(self.layers).use_cosine_sim, **mlp_kwargs) for _ in range(num_quantizers - 1)])
        else:
            self.mlps = (None,) * (num_quantizers - 1)

        # sharing codebook logic

        self.shared_codebook = shared_codebook

        if not shared_codebook:
            return

        assert self.uniform_codebook_size

        first_vq, *rest_vq = self.layers
        codebook = first_vq._codebook
        # codebook = first_vq.embedding
        
        for vq in rest_vq:
            vq._codebook = codebook
            # vq.embedding = codebook
    
    @property
    def codebook_size(self):
        return self.layers[0].codebook_size
    
    @property
    def codebook_dim(self):
        return self.layers[0].codebook_dim

    @property
    def codebooks(self):
        codebooks = [layer._codebook.embed for layer in self.layers]

        codebooks = tuple(rearrange(codebook, '1 ... -> ...') for codebook in codebooks)

        if not self.uniform_codebook_size:
            return codebooks

        codebooks = torch.stack(codebooks)
        return codebooks

    def get_codes_from_indices(self, indices):

        batch, quantize_dim = indices.shape[0], indices.shape[-1]

        # may also receive indices in the shape of 'b h w q' (accept_image_fmap)

        indices, ps = pack([indices], 'b * q')

        # because of quantize dropout, one can pass in indices that are coarse
        # and the network should be able to reconstruct

        if quantize_dim < self.num_quantizers:
            assert self.quantize_dropout > 0., 'quantize dropout must be greater than 0 if you wish to reconstruct from a signal with less fine quantizations'
            indices = F.pad(indices, (0, self.num_quantizers - quantize_dim), value = -1)

        # take care of quantizer dropout

        mask = indices == -1.
        indices = indices.masked_fill(mask, 0) # have it fetch a dummy code to be masked out later

        if not self.implicit_neural_codebook and self.uniform_codebook_size:

            all_codes = get_at('q [c] d, b n q -> q b n d', self.codebooks, indices)

        else:
            # else if using implicit neural codebook, or non uniform codebook sizes, codes will need to be derived layer by layer

            code_transform_mlps = (None, *self.mlps)

            all_codes = []
            quantized_out = 0.

            for codes, indices, maybe_transform_mlp in zip(self.codebooks, indices.unbind(dim = -1), code_transform_mlps):

                if exists(maybe_transform_mlp):
                    codes = maybe_transform_mlp(codes, condition = quantized_out)
                    layer_codes = get_at('b n [c] d, b n -> b n d', codes, indices)
                else:
                    layer_codes = get_at('[c] d, b n -> b n d', codes, indices)

                all_codes.append(layer_codes)
                quantized_out += layer_codes

            all_codes = torch.stack(all_codes)

        # mask out any codes that were dropout-ed

        all_codes = all_codes.masked_fill(rearrange(mask, 'b n q -> q b n 1'), 0.)

        # if (accept_image_fmap = True) then return shape (quantize, batch, height, width, dimension)

        all_codes, = unpack(all_codes, ps, 'q b * d')

        return all_codes

    def get_output_from_indices(self, indices):
        codes = self.get_codes_from_indices(indices)
        codes_summed = reduce(codes, 'q ... -> ...', 'sum')
        return self.project_out(codes_summed)

    def forward(
        self,
        x,
        mask = None,
        indices: Tensor | list[Tensor] | None = None,
        return_all_codes = False,
        sample_codebook_temp = None,
        freeze_codebook = False,
        rand_quantize_dropout_fixed_seed = None
    ):
        num_quant, quant_dropout_multiple_of, return_loss, device = self.num_quantizers, self.quantize_dropout_multiple_of, exists(indices), x.device

        x = self.project_in(x)

        assert not (self.accept_image_fmap and exists(indices))

        quantized_out = 0.
        residual = x

        all_losses = []
        all_indices = []
        all_perplexity = []

        if isinstance(indices, list):
            indices = torch.stack(indices)

        if return_loss:
            assert not torch.any(indices == -1), 'some of the residual vq indices were dropped out. please use indices derived when the module is in eval mode to derive cross entropy loss'
            ce_losses = []

        should_quantize_dropout = self.training and self.quantize_dropout and not return_loss

        # sample a layer index at which to dropout further residual quantization
        # also prepare null indices and loss

        if should_quantize_dropout:

            # check if seed is manually passed in

            if not exists(rand_quantize_dropout_fixed_seed):
                rand_quantize_dropout_fixed_seed = get_maybe_sync_seed(device)

            rand = random.Random(rand_quantize_dropout_fixed_seed)

            rand_quantize_dropout_index = rand.randrange(self.quantize_dropout_cutoff_index, num_quant)

            if quant_dropout_multiple_of != 1:
                rand_quantize_dropout_index = round_up_multiple(rand_quantize_dropout_index + 1, quant_dropout_multiple_of) - 1

            null_indices_shape = (x.shape[0], *x.shape[-2:]) if self.accept_image_fmap else tuple(x.shape[:2])
            null_indices = torch.full(null_indices_shape, -1., device = device, dtype = torch.long)
            null_loss = torch.full((1,), 0., device = device, dtype = x.dtype)

        # setup the mlps for implicit neural codebook

        maybe_code_transforms = (None,) * len(self.layers)

        if self.implicit_neural_codebook:
            maybe_code_transforms = (None, *self.mlps)

        # save all inputs across layers, for use during expiration at end under shared codebook setting

        all_residuals = []

        # go through the layers

        for quantizer_index, (vq, maybe_mlp) in enumerate(zip(self.layers, maybe_code_transforms)):

            if should_quantize_dropout and quantizer_index > rand_quantize_dropout_index:
                all_indices.append(null_indices)
                all_losses.append(null_loss)
                continue

            layer_indices = None
            if return_loss:
                layer_indices = indices[..., quantizer_index]

            # setup the transform code function to be passed into VectorQuantize forward

            if exists(maybe_mlp):
                maybe_mlp = partial(maybe_mlp, condition = quantized_out)

            # save for expiration

            all_residuals.append(residual)

            # vector quantize forward

            # quantized, *rest = vq(
            #     residual,
            #     mask = mask,
            #     indices = layer_indices,
            #     sample_codebook_temp = sample_codebook_temp,
            #     freeze_codebook = freeze_codebook,
            #     codebook_transform_fn = maybe_mlp
            # )
            quantized, *rest = vq(
                residual,
            )

            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized

            if return_loss:
                ce_loss = rest[0]
                ce_losses.append(ce_loss)
                continue

            # embed_indices, loss = rest
            loss, (perplexity, min_encodings, embed_indices) = rest

            all_indices.append(embed_indices)
            all_losses.append(loss)
            all_perplexity.append(perplexity)

        # if shared codebook, update ema only at end

        if self.training and self.shared_codebook:
            shared_layer = first(self.layers)
            shared_layer._codebook.update_ema()
            shared_layer.update_in_place_optimizer()
            shared_layer.expire_codes_(torch.cat(all_residuals, dim = -2))

        # project out, if needed

        quantized_out = self.project_out(quantized_out)

        # whether to early return the cross entropy loss

        if return_loss:
            return quantized_out, sum(ce_losses)

        # stack all losses and indices

        all_losses, all_indices, all_perplexity = map(partial(torch.stack, dim = -1), (all_losses, all_indices, all_perplexity))
        ret = (quantized_out, all_losses.sum(), all_indices, all_perplexity)
        # all_losses, all_indices = map(partial(torch.stack, dim = -1), (all_losses, all_indices))
        # ret = (quantized_out, all_losses.sum(), all_indices)

        if return_all_codes:
            # whether to return all codes from all codebooks across layers
            all_codes = self.get_codes_from_indices(all_indices)

            # will return all codes in shape (quantizer, batch, sequence length, codebook dimension)
            ret = (*ret, all_codes)

        return ret[0], ret[1], (ret[2:])

# grouped residual vq

class GroupedResidualVQ(Module):
    def __init__(
        self,
        *,
        dim,
        groups = 1,
        accept_image_fmap = False,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.groups = groups
        assert (dim % groups) == 0
        dim_per_group = dim // groups

        self.accept_image_fmap = accept_image_fmap

        self.rvqs = ModuleList([])

        for _ in range(groups):
            self.rvqs.append(ResidualVQ(
                dim = dim_per_group,
                accept_image_fmap = accept_image_fmap,
                **kwargs
            ))

    @property
    def codebooks(self):
        return torch.stack(tuple(rvq.codebooks for rvq in self.rvqs))

    @property
    def split_dim(self):
        return 1 if self.accept_image_fmap else -1

    def get_codes_from_indices(self, indices):
        codes = tuple(rvq.get_codes_from_indices(chunk_indices) for rvq, chunk_indices in zip(self.rvqs, indices))
        return torch.stack(codes)

    def get_output_from_indices(self, indices):
        outputs = tuple(rvq.get_output_from_indices(chunk_indices) for rvq, chunk_indices in zip(self.rvqs, indices))
        return torch.cat(outputs, dim = self.split_dim)

    def forward(
        self,
        x,
        indices = None,
        return_all_codes = False,
        sample_codebook_temp = None,
        freeze_codebook = False,
        mask = None,
    ):
        shape, split_dim, device = x.shape, self.split_dim, x.device
        assert shape[split_dim] == self.dim

        # split the feature dimension into groups

        x = x.chunk(self.groups, dim = split_dim)

        indices = default(indices, tuple())
        return_ce_loss = len(indices) > 0
        assert len(indices) == 0 or len(indices) == self.groups

        forward_kwargs = dict(
            return_all_codes = return_all_codes,
            sample_codebook_temp = sample_codebook_temp,
            mask = mask,
            freeze_codebook = freeze_codebook,
            rand_quantize_dropout_fixed_seed = get_maybe_sync_seed(device) if self.training else None
        )

        # invoke residual vq on each group

        out = tuple(rvq(chunk, indices = chunk_indices, **forward_kwargs) for rvq, chunk, chunk_indices in zip_longest(self.rvqs, x, indices))
        out = tuple(zip(*out))

        # if returning cross entropy loss to rvq codebooks

        if return_ce_loss:
            quantized, ce_losses = out
            return torch.cat(quantized, dim = split_dim), sum(ce_losses)

        # otherwise, get all the zipped outputs and combine them

        quantized, all_indices, commit_losses, *maybe_all_codes = out

        quantized = torch.cat(quantized, dim = split_dim)
        all_indices = torch.stack(all_indices)
        commit_losses = torch.stack(commit_losses)

        ret = (quantized, all_indices, commit_losses, *maybe_all_codes)
        return ret
