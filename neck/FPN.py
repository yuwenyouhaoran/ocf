import torch
import torch.nn as nn


class PyramidFeatures(nn.Module):
    def __init__(self,C2_size, C3_size, C4_size, C5_size,C6_size, feature_size=256):
        super(PyramidFeatures, self).__init__()
        # TODO  simple FPN
        # upsample C5 to get P5 from the FPN paper
        self.P6_1 = nn.Conv2d(C6_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P6_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        # self.P6_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        # self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        # self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        # self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P3 elementwise to C3
        self.P2_1 = nn.Conv2d(C2_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P2_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P2_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

          # add P3 elementwise to C3
        self.P1_1 = nn.Conv2d(64, feature_size, kernel_size=1, stride=1, padding=0)
        self.P1_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P1_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # # "P6 is obtained via a 3x3 stride-2 conv on C5"
        # self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)
        #
        # # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        # self.P7_1 = nn.ReLU()
        # self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        C1,C2, C3, C4, C5, C6 = inputs

        P6_x = self.P6_1(C6)
        P6_upsampled_x = self.P6_upsampled(P6_x)
        # P6_x = self.P6_2(P6_x)

        P5_x = self.P5_1(C5)
        P5_x = P6_upsampled_x + P5_x
        P5_upsampled_x = self.P5_upsampled(P5_x)
        # P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        # P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P4_upsampled_x + P3_x
        P3_upsampled_x = self.P3_upsampled(P3_x)
        # P3_x = self.P3_2(P3_x)

        P2_x = self.P2_1(C2)
        P2_x = P3_upsampled_x + P2_x
        P2_upsampled_x = self.P2_upsampled(P2_x)
        # P2_x = self.P2_2(P2_x)
     
        P1_x = self.P1_1(C1)
        P1_x = P2_upsampled_x + P1_x
        P1_upsampled_x = self.P1_upsampled(P1_x)


        return P1_upsampled_x
