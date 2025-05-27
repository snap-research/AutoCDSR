# Revisiting Self-attention for Cross-domain Sequential Recommendation

Source code for the paper **[Revisiting Self-attention for Cross-domain Sequential Recommendation]()** accepted at KDD 2025.
>by Clark Mingxuan Ju, Leonardo Neves, Bhuvesh Kumar, Liam Collins, Tong Zhao, Yuwei Qiu, Qing Dou, Sohail Nizam, Sen Yang, and Neil Shah.

The paper proposes AutoCDSR which utilizes vanilla self-attention transformers to achieve good performance for cross-domain sequential recommendation. 

## 1. Installation

Please install all dependencies using the command:
```
conda create --name <env> --file requirements.txt
```


## 2. Run experiments
This is one example of reproducing results for KuaiRand-1K. 

To train AutoCDSR+ model with BERT4Rec, run:

```bash
python src/train.py trainer=ddp experiment=kuairand model=hf_transformer_cd_sid_ib_kuairand_pareto
```

To train BERT4Rec model, run:

```bash
python src/train.py trainer=ddp experiment=kuairand model=hf_transformer_kuairand
```

## 3. Reference
If you find this repo and our work useful to you, please kindly cite us using:

```
@article{ju2025revisiting,
  title={Revisiting Self-attention for Cross-domain Sequential Recommendation},
  author={Ju, Clark Mingxuan and Neves, Leonardo and Kumar, Bhuvesh and Collins, Liam and Zhao, Tong and Qiu, Yuwei and Dou, Ching and Nizam, Sohail and Yang, Sen and Shah, Neil},
  journal={Proceedings of the 31st ACM SIGKDD conference on knowledge discovery and data mining},
  year={2025}
}
```

## 4. Contact
Please contact mju@snap.com for any questions.
