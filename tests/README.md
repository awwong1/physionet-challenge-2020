Test data files for evaluation helper functions.

```bash
# train a model
$ python3 driver.py tests/model/ tests/data/ tests/output/
# cd into evaluation-2020, using tests/model/finalized_model_1594678260.sav
$ python3 evaluate_12ECG_score.py ../tests/data ../tests/output
```
```text
Finding label and output files...
Loading labels and outputs...
Organizing labels and outputs...
Loading weights...
Evaluating model...
- AUROC and AUPRC...
- Accuracy...
- F-measure...
- F-beta and G-beta measures...
- Challenge metric...
Done.
AUROC,AUPRC,Accuracy,F-measure,Fbeta-measure,Gbeta-measure,Challenge metric
0.891,0.735,0.393,0.563,0.605,0.444,0.635
```

## Files

```text
SCORED
427084000 ['STach', 'sinus tachycardia'] data/Training_2/Q0001
427084000 ['STach', 'sinus tachycardia'] data/Training_2/Q0022
427172004 ['PVC', 'premature ventricular contractions'] data/Training_2/Q0010
426627000 ['Brady', 'bradycardia'] data/Training_2/Q0012
63593006 ['SVPB', 'supraventricular premature beats'] data/Training_2/Q0017
284470004 ['PAC', 'premature atrial contraction'] data/Training_2/Q0026
713427006 ['CRBBB', 'complete right bundle branch block'] data/Training_2/Q0027
164909002 ['LBBB', 'left bundle branch block'] data/Training_2/Q0033
164889003 ['AF', 'atrial fibrillation'] data/Training_2/Q0048
270492004 ['IAVB', '1st degree av block'] data/Training_2/Q0050
713426002 ['IRBBB', 'incomplete right bundle branch block'] data/Training_2/Q0077
164890007 ['AFL', 'atrial flutter'] data/Training_2/Q0096
164934002 ['TAb', 't wave abnormal'] data/Training_2/Q0122
10370003 ['PR', 'pacing rhythm'] data/Training_2/Q0398
17338001 ['VEB', 'ventricular ectopic beats'] data/Training_2/Q0630
59931005 ['TInv', 't wave inversion'] data/Training_2/Q1240
426783006 ['SNR', 'sinus rhythm'] data/Training_2/Q1804
59118001 ['RBBB', 'right bundle branch block'] data/Training_2/Q1805
426177001 ['SB', 'sinus bradycardia'] data/Training_2/Q1808
427393009 ['SA', 'sinus arrhythmia'] data/Training_2/Q1847
47665007 ['RAD', 'right axis deviation'] data/Training_2/Q1850
111975006 ['LQT', 'prolonged qt interval'] data/Training_2/Q1917
698252002 ['NSIVCB', 'nonspecific intraventricular conduction disorder'] data/Training_2/Q2428
39732003 ['LAD', 'left axis deviation'] data/WFDB/E00009
251146004 ['LQRSV', 'low qrs voltages'] data/WFDB/E00015
445118002 ['LAnFB', 'left anterior fascicular block'] data/WFDB/E00018
164917005 ['QAb', 'qwave abnormal'] data/WFDB/E00793
164947007 ['LPR', 'Prolonged PR interval'] data/WFDB/HR00599
OTHER
164861001 ['MIs', 'myocardial ischemia'] data/WFDB/HR20715
```