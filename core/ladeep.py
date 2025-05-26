import torch
import torch.nn as nn
from .cle_and_cld import CharacteristicLineEncoder, CharacteristicLineDecoder
from .cse_and_csd import CrossSectionEncoder, CrossSectionDecoder
from .off import ObjectFeatureFusioner
from .mpe import MotionParametersEncoder
from .dp import LoadingModule, UnloadingModule


class LaDEEP(nn.Module):
    def __init__(
            self,
            input_dim = 3,
            input_length = 300,
            seq_length = 60,
            emb_dim = 64,
            output_length = 300,
            loading_layers = 3,
            loading_heads = 4,
            loading_fea_len = 64,
            unloading_layers = 3,
            unloading_heads = 4,
            unloading_fea_len = 64,
            motion_degrees = 6
    ):
        super().__init__()
        self.cle_for_mould = CharacteristicLineEncoder(
            dim = input_dim,
            input_length = input_length,
            seq_length = seq_length,
            emb_dim = emb_dim
        )
        self.cle_for_strip = CharacteristicLineEncoder(
            dim = input_dim,
            input_length = input_length,
            seq_length = seq_length,
            emb_dim = emb_dim
        )
        self.cse = CrossSectionEncoder()
        self.csd = CrossSectionDecoder()
        self.off = ObjectFeatureFusioner(cl_dim = emb_dim)
        self.mpe = MotionParametersEncoder(
            seq_length = seq_length,
            emb_dim = emb_dim,
            motion_degrees = motion_degrees
        )
        self.dp_loading = LoadingModule(
            num_layers = loading_layers,
            num_heads = loading_heads,
            fea_len = loading_fea_len
        )
        self.dp_unloading = UnloadingModule(
            num_layers = unloading_layers,
            num_heads = unloading_heads,
            fea_len = unloading_fea_len
        )
        self.cld_for_loading = CharacteristicLineDecoder(
            dim = input_dim,
            seq_length = seq_length,
            emb_dim = emb_dim,
            output_length = output_length
        )
        self.cld_for_unloading = CharacteristicLineDecoder(
            dim = input_dim,
            seq_length = seq_length,
            emb_dim = emb_dim,
            output_length = output_length
        )
        self._initialize()

    def forward(self, strip, mould, section, params):
        strip = self.cle_for_strip(strip)
        mould = self.cle_for_mould(mould)
        section = self.cse(section)
        recover_section = self.csd(section)
        strip = self.off(strip, section)
        params = self.mpe(params)
        strip = self.dp_loading(params, mould, strip)
        loaded_strip = self.cld_for_loading(strip)
        strip = self.dp_unloading(strip)
        unloaded_strip = self.cld_for_unloading(strip)

        return recover_section, loaded_strip, unloaded_strip

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d)):
                if m.weight.requires_grad:
                    nn.init.kaiming_normal_(m.weight, mode = "fan_out", nonlinearity = "relu")
                if m.bias is not None and m.bias.requires_grad:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if m.weight.requires_grad:
                    nn.init.xavier_uniform_(m.weight, gain = nn.init.calculate_gain("relu"))
                if m.bias is not None and m.bias.requires_grad:
                    nn.init.constant_(m.bias, 0)
        return True
