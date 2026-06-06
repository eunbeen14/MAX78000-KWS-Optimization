# ==============================================================================
# [Custom Modified for KWS20 Optimization]
# Note: Designed to maximize the Receptive Field for long-term time-series patterns.
# Dropped from the final candidate list due to the physical padding limitation (max 2) 
# of the MAX78000 NPU, which conflicts with the padding=4 required for dilation.
# ==============================================================================

from torch import nn
import ai8x

class AI87KWS20Net_TCN(nn.Module):
    def __init__(
            self,
            num_classes=21,
            num_channels=128,
            dimensions=(128, 1),
            bias=False,
            **kwargs
    ):
        super().__init__()
        self.drop = nn.Dropout(p=0.2)
        
        self.voice_conv1 = ai8x.FusedConv1dReLU(num_channels, 100, 1, stride=1, padding=0, bias=bias, **kwargs)
        self.voice_conv2 = ai8x.FusedConv1dReLU(100, 96, 3, stride=1, padding=0, bias=bias, **kwargs)
        self.voice_conv3 = ai8x.FusedMaxPoolConv1dReLU(96, 64, 3, stride=1, padding=0, bias=bias, **kwargs)
        self.voice_conv4 = ai8x.FusedConv1dReLU(64, 48, 3, stride=1, padding=0, bias=bias, **kwargs)
        
        # [Hardware Constraint] MAX78000 hardware padding limit is 2. 
        # Attempting padding=4 with dilation causes Synthesis AssertionError.
        self.kws_conv1 = ai8x.FusedConv1dReLU(48, 64, 3, stride=1, padding=2,
                                              dilation=2, bias=bias, **kwargs)
        self.kws_conv2 = ai8x.FusedConv1dReLU(64, 96, 3, stride=1, padding=4,
                                              dilation=4, bias=bias, **kwargs)
        
        self.kws_conv3 = ai8x.FusedAvgPoolConv1dReLU(96, 100, 3, stride=1, padding=0, bias=bias, **kwargs)
        self.kws_conv4 = ai8x.FusedMaxPoolConv1dReLU(100, 64, 3, stride=1, padding=0, bias=bias, **kwargs)
        
        self.fc = ai8x.Linear(192, num_classes, bias=bias, wide=True, **kwargs)

    def forward(self, x):
        x = self.voice_conv1(x)
        x = self.voice_conv2(x)
        x = self.drop(x)
        x = self.voice_conv3(x)
        x = self.voice_conv4(x)
        x = self.drop(x)
        x = self.kws_conv1(x)
        x = self.kws_conv2(x)
        x = self.drop(x)
        x = self.kws_conv3(x)
        x = self.kws_conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def ai87kws20net_tcn(pretrained=False, **kwargs):
    assert not pretrained
    return AI87KWS20Net_TCN(**kwargs)

models = [
    {
        'name': 'ai87kws20net_tcn',
        'min_input': 1,
        'dim': 1,
    },
]