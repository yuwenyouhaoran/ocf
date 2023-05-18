# occupancy_flow_predict

train occupancy_flow_predict
```
cd occupancy_flow_predict
python -m torch.distributed.launch --nproc_per_node 4 train_orgin.py
```
visualization
```
python model_analysis.py
```
test
```
python test.py
```
data processing

```
cd data
python data_process.py  
python data_process_origin.py 
```

file
```
-data 
--data_process.py  
--data_process_origin.py 
-head
--convlstm.py 
-losses
--waymo_loss.py 
-neck
--BiFPN.py
--FPN.py 
-network


```
reference

(https://github.com/ndrplz/ConvLSTM_pytorch)


(https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/blob/c533bc2de65135a6fe1d25ca437765c630943afb/efficientdet/model.py#L55)

