# ==============================================================================
# [Custom Modified for KWS20 Optimization]
# Note: This DS-CNN model was designed to maximize memory efficiency. 
# However, it serves as a negative result (Hardware-Algorithm Mismatch) because
# the MAX78000 NPU is hardwired for dense convolutions. The depthwise convolution 
# (groups=96) triggers an AssertionError during hardware synthesis.
# ==============================================================================

from torch import nn
import ai8x

class AI87KWS20Net_DSCNN(nn.Module):
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
        
        # [Hardware Constraint] MAX78000 does not support depthwise routing. 
        # This groups=96 setup causes a synthesis error (supported from MAX78002).
        self.voice_conv3_dw = ai8x.FusedMaxPoolConv1dReLU(96, 96, 3, stride=1, padding=1,
                                                          groups=96, bias=bias, **kwargs)
        self.voice_conv3_pw = ai8x.FusedConv1dReLU(96, 64, 1, stride=1, padding=0,
                                                   bias=bias, **kwargs)
        
        self.voice_conv4 = ai8x.FusedConv1dReLU(64, 48, 3, stride=1, padding=0, bias=bias, **kwargs)
        self.kws_conv1 = ai8x.FusedMaxPoolConv1dReLU(48, 64, 3, stride=1, padding=0, bias=bias, **kwargs)
        self.kws_conv2 = ai8x.FusedConv1dReLU(64, 96, 3, stride=1, padding=0, bias=bias, **kwargs)
        self.kws_conv3 = ai8x.FusedAvgPoolConv1dReLU(96, 100, 3, stride=1, padding=0, bias=bias, **kwargs)
        self.kws_conv4 = ai8x.FusedMaxPoolConv1dReLU(100, 64, 3, stride=1, padding=0, bias=bias, **kwargs)
        
        self.fc = ai8x.Linear(192, num_classes, bias=bias, wide=True, **kwargs)

    def forward(self, x):
        x = self.voice_conv1(x)
        x = self.voice_conv2(x)
        x = self.drop(x)
        
        x = self.voice_conv3_dw(x)
        x = self.voice_conv3_pw(x)
        
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

def ai87kws20net_dscnn(pretrained=False, **kwargs):
    assert not pretrained
    return AI87KWS20Net_DSCNN(**kwargs)

models = [
    {
        'name': 'ai87kws20net_dscnn',
        'min_input': 1,
        'dim': 1,
    },
]