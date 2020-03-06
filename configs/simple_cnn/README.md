SimpleCNN configurations here use 5-fold cross validation.

```bash
# make sure that main.py is in the current working directory

./configs/simple_cnn/run_cv5.sh  # this assumes NVIDIA/apex is installed
```

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
