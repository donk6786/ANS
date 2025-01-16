# ANS-Rechorus
## 基于[Rechorus](https://github.com/THUwangcy/ReChorus)实现[ANS](https://dl.acm.org/doi/pdf/10.1145/3690656)算法的复现，并且与框架中的同类型算法在不止一个数据集上进行对比。
### Getting Started

```bash
git clone https://github.com/donk6786/ANS.git
#主要修改了Rechorus框架的baserunner和main，以及写入了ANS.py文件
```

在**ANS**目录下创建虚拟环境

```bash
pip install -r requirements.txt
```
  
运行模型

```bash
python src\main.py --model_name ANS --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset Grocery_and_Gourmet_Food 
```

运行结果 \(与**BPRMF**以及**LightGCN**算法对比, 运行参数相同 \(若有同种参数, e.g: batch_size\) \)
先展示训练后dev的效果
|Dataset                     |Method  |HR@5   |NDCG@5 |HR@10  |NDCG@10|HR@20  |NDCG@20|HR@50  |NDCG@50|
|:---:                       |:---:   |:---:  |:---:  |:---:  |:---:  |:---:  |:---:  |:---:  |:---:  |
|                            |ANS     |0.4043 |0.2915 |0.5041 |0.3238 |0.6098 |0.3505 |0.8027 |0.3885 |
|**Grocery_and_Gourmet_Food**|BPRMF   |0.4150 |0.3022 |0.5147 |0.3346 |0.6186 |0.3608 |0.8026 |0.3971 |
|                            |LightGCN|0.4669 |0.3369 |0.5837 |0.3750 |0.6872 |0.4011 |0.8566 |0.4346 |
|                            |        |       |       |       |       |       |       |       |       |
|                            |ANS     |0.0543 |0.0320 |0.1202 |0.0531 |0.2411 |0.0834 |0.5218 |0.1386 |
|**MIND_Large**              |BPRMF   |0.0641 |0.0385 |0.1291 |0.0591 |0.2362 |0.0860 |0.5325 |0.1441 |
|                            |LightGCN|0.0574 |0.0336 |0.1288 |0.0565 |0.2610 |0.0897 |0.5451 |0.1457 |
再展示训练后test的效果
|Dataset                     |Method  |HR@5   |NDCG@5 |HR@10  |NDCG@10|HR@20  |NDCG@20|HR@50  |NDCG@50|
|                            |ANS     |0.3357 |0.2320 |0.4387 |0.2654 |0.5514 |0.2938 |0.7572 |0.3343 |
|**Grocery_and_Gourmet_Food**|BPRMF   |0.3460 |0.2393 |0.4545 |0.2746 |0.5638 |0.3021 |0.7667 |0.3420 |
|                            |LightGCN|0.3918 |0.2718 |0.5160 |0.3124 |0.6265 |0.3403 |0.8288 |0.3802 |
|                            |        |       |       |       |       |       |       |       |       |
|                            |ANS     |0.1029 |0.0656 |0.1686 |0.0868 |0.3020 |0.1202 |0.5069 |0.1609 |
|**MIND_Large**              |BPRMF   |0.1108 |0.0723 |0.1794 |0.0947 |0.2922 |0.1229 |0.5608 |0.1757 |
|                            |LightGCN|0.1098 |0.0649 |0.1784 |0.0874 |0.2931 |0.1161 |0.5402 |0.1646 |
