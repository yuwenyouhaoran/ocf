# occupancy_flow_predict

训练 occupancy_flow_predict
```
cd occupancy_flow_predict
python -m torch.distributed.launch --nproc_per_node 4 train_orgin.py
```
可视化
```
python model_analysis.py
```
test(note:tf转pytorch后数据可能存在问题)
```
python test.py
```
数据预处理

```
cd data
python data_process.py  #生成通道数为98的数据，具体参考报告
python data_process_origin.py #生成的是通道数为23的数据，其中只有occupancy和道路总体图
```

文件描述
```
-data 
--data_process.py  #生成通道数为98的数据，具体参考报告
--data_process_origin.py #生成的是通道数为23的数据，其中只有occupancy和道路总体图
-head
--convlstm.py #解码器部分，具体原理参考报告
-losses
--waymo_loss.py #loss函数部分
-neck
#聚合器部分
--BiFPN.py
--FPN.py 
-network
#此部分为各种backbone的具体实现

```
参考

[convlstm实现](https://github.com/ndrplz/ConvLSTM_pytorch)


[BiFPN实现](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/blob/c533bc2de65135a6fe1d25ca437765c630943afb/efficientdet/model.py#L55)

