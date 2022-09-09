import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn as nn
from network.regnet import regnetx_320
from network.swinV2 import build_swinv2
from neck.bifpn import BiFPN
from neck.FPN import PyramidFeatures
from head.convlstm import ConvLSTM
# from torchsummary import summary
from torch.utils.checkpoint import checkpoint
# from torch.profiler import profile, record_function, ProfilerActivity
class Hope(nn.Module):
    def __init__(self):
        super().__init__()
        # self.model = regnetx_320(pretrained=True)
        self.model=build_swinv2()
        
        compound_coef = 0
        fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384, 384]
        fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8, 8]
        conv_channel_coef = {
            # the channels of P3/P4/P5.
            # 0: [40, 112, 320],
            0: [96, 192, 384, 768, 1536],
            1: [40, 112, 320],
            2: [48, 120, 352],
            3: [48, 136, 384],
            4: [56, 160, 448],
            5: [64, 176, 512],
            6: [72, 200, 576],
            7: [72, 200, 576],
            8: [80, 224, 640],
        }
        self.neck = nn.Sequential(
            *[BiFPN(fpn_num_filters[compound_coef],
                    conv_channel_coef[compound_coef],
                    True if _ == 0 else False,
                    attention=True if compound_coef < 6 else False,
                    use_p8=compound_coef > 7)
              for _ in range(fpn_cell_repeats[compound_coef])])
        self.fpn = PyramidFeatures(C2_size=64, C3_size=64, C4_size=64, C5_size=64, C6_size=64)
        self.clstm = ConvLSTM(input_dim=256,
                              hidden_dim=[64, 64, 64],
                              kernel_size=(3, 3),
                              num_layers=3,
                              batch_first=True,
                              bias=True,
                              return_all_layers=False)
        self.upsample = nn.Upsample(scale_factor=2)
        self.shared_head = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=0)

        # decoder_channels = [32, 64, 128, 256, 512]

        # self.decoder=nn.ModuleList()
        # for i in [4,3,2,1,0]:
        #     if i==4:
        #         in_channels=1536
        #     else:
        #         in_channels=decoder_channels[i+1]
        #     sub_decoder=nn.Sequential(
        #         nn.Conv2d(in_channels=in_channels,out_channels=decoder_channels[i],kernel_size=3,padding=1),
        #         nn.ReLU(),
        #         nn.Upsample(scale_factor=2),
        #         nn.Conv2d(in_channels=decoder_channels[i],out_channels=decoder_channels[i],kernel_size=3,padding=1),
        #         nn.ReLU()

        #     )
        #     self.decoder.append(sub_decoder)
        # self.outlayer=nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding=1)
    def forward(self, x):
        # x = self.model(x)
        feature1, feature2, feature3, feature4, feature5 = self.model(x)
        # print('p3',feature1.shape)
        # print('p4',feature2.shape)
        # print('p5',feature3.shape)
        # print('p6',feature4.shape)
        # print('p7',feature5.shape)

        features = (feature1, feature2, feature3, feature4,feature5)
        P = self.neck(features)
        # P1 = P[0]
        # P2_6 = (P1,P[1], P[2], P[3], P[4], P[5])
        # P2 = self.fpn(P2_6)
        output = self.shared_head(P[0])
        output=self.upsample(output)
        output = output.permute(0, 2, 3, 1).contiguous()
        # P2 = torch.unsqueeze(P2, dim=1).contiguous()
        # output_list, state_list = self.clstm(input_tensor=P2, hidden_state=P1)
        # output = 0
        # for i in range(len(output_list)):
        #     output = output_list[i].squeeze(dim=1)
        #     output = self.upsample(output)
        #     output = self.shared_head(output)
        #     output = output.permute(0, 2, 3, 1).contiguous()
        return output

        # for i in range(len(self.decoder)):
        #     x=self.decoder[i](x)
        # x=self.outlayer(x).permute(0,2,3,1).contiguous()
        # return x

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    initial_usage = torch.cuda.memory_allocated()
    print("0", initial_usage)  # 0

# 模型初始化
    model=build_swinv2().cuda()

    inputs = torch.randn(size=(1, 23, 768,768), device="cuda:0")
    # innputs_usage = torch.cuda.memory_allocated()
    # print("innputs_usage", innputs_usage/1024/1024)  # 0
    model=Hope()
    model=model.cuda()
    # model_usage = torch.cuda.memory_allocated()
    # print("model_usage", (model_usage-innputs_usage)/1024/1024)  # 0
    output=model(inputs)
    
    # forward_usage = torch.cuda.memory_allocated()
    # print("forward_usage", (forward_usage-model_usage)/1024/1024)  # 0
    # loss=torch.sum(output)
    # loss_usage = torch.cuda.memory_allocated()
    # print("loss_usage", (loss_usage-forward_usage)/1024/1024)
    # loss.backward()
    # backward_usage = torch.cuda.memory_allocated()
    # print("backward_usage", (backward_usage-loss_usage)/1024/1024)