# ds-lab
2019 Data Science Lab Project. Team: Casamento, Palumbo, Stolfo.


## Files description
- Age discrepancy_t_tests: computes t-tests pvalues for the mean of age discrepancy in patients with 4 of the most frequent disorders versus healthy patients.
- AgeDiscrepancy_controllingForAge: computes pvalues for the mean of age discrepancy in patients with 4 of the most frequent disorders versus healthy patients, controlling for age.
- AgeHealthy: tests on quality of age prediction on healthy patients only (CDE versus XGBoost).
- Age_EEG: Age prediction from EEG data (average+cluster features). Every model is fine-tuned with grid search.
- Age_MRI: Age prediction from MRI data (structural only and structural+DTI). Every model is fine-tuned with grid search.
- Age_EEG_MRI: Age prediction from MRI and EEG combined. Every model is fine-tuned with grid search.
- CDE: age prediction with Conditional Density Estimation model.
- FindTestset: A random subset of 120 patients is selected as testset. The corresponding test indices are found in the 'data' folder, named ''


## Useful links
- [EEG data](https://www.dropbox.com/sh/xsevywkt1tjgy0c/AAAWiFH73fNXV9wNc33ETcuOa?dl=0)

- [GoogleDOCS](https://docs.google.com/document/d/1tIK002Oo6fbXMcbpMgvnt9_lJ3M1XI1ETATAdRQIo-M/edit)

- [Paper 1]( https://arxiv.org/abs/1903.00954)

- [Paper 2](https://arxiv.org/pdf/1907.08982.pdf)

- [CDE code](https://github.com/freelunchtheorem/Conditional_Density_Estimation)


