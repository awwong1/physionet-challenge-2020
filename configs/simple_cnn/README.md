SimpleCNN configurations here use 5-fold cross validation.

```python
t_rec, v_rec = PhysioNet2020Dataset.split_names_cv("Training_WFDB", 5, 0)
# t_rec == ("A1377", ..., "A6877")
# v_rec == ("A0001", ..., "A1376")

t_rec, v_rec = PhysioNet2020Dataset.split_names_cv("Training_WFDB", 5, 1)
# t_rec == ("A0001", ..., "A6877")
# v_rec == ("A1377", ..., "A2752")

t_rec, v_rec = PhysioNet2020Dataset.split_names_cv("Training_WFDB", 5, 4)
# t_rec == ("A0001", ..., "A5504")
# v_rec == ("A5505", ..., "A6877")
```

```bash
# make sure that main.py is in the current working directory
./configs/simple_cnn/run_cv5.sh  # this assumes NVIDIA/apex is installed
```
```text
SimpleCNN/cv5-0 results:
AUROC|AUPRC|Accuracy|F-measure|Fbeta-measure|Gbeta-measure
0.431|0.223|0.852|0.391|0.448|0.237
SimpleCNN/cv5-1 results:
AUROC|AUPRC|Accuracy|F-measure|Fbeta-measure|Gbeta-measure
0.534|0.253|0.824|0.281|0.298|0.148
SimpleCNN/cv5-2 results:
AUROC|AUPRC|Accuracy|F-measure|Fbeta-measure|Gbeta-measure
0.463|0.247|0.817|0.285|0.336|0.155
SimpleCNN/cv5-3 results:
AUROC|AUPRC|Accuracy|F-measure|Fbeta-measure|Gbeta-measure
0.473|0.246|0.833|0.342|0.378|0.198
SimpleCNN/cv5-4 results:
AUROC|AUPRC|Accuracy|F-measure|Fbeta-measure|Gbeta-measure
0.442|0.218|0.850|0.379|0.409|0.217
```

Full output files are available at [experiments/PhysioNet2020/SimpleCNN-cv5.tar.gz](https://swift-yeg.cloud.cybera.ca:8080/v1/AUTH_e3b719b87453492086f32f5a66c427cf/physionet_2020/experiments/PhysioNet2020/SimpleCNN-cv5.tar.gz)
