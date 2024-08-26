# EEG-pyline_object: EEG pipeline in Python
[![DOI](https://zenodo.org/badge/495300654.svg)](https://zenodo.org/badge/latestdoi/495300654)

This is an objected oriented version of pyeline created by Jules Mitchell using the original code from Toomas Erik Anijärv.

The main aim for creating the version was to advance the pipelines goalof making EEG analysis in Python easier for other researchers who are not too familiar with programming but also do not want to use other commercial blackbox-style software. Using ready-made Jupyter notebooks, it is easy to get started with EEG data pre-processing, spectral analysis, and ERP analysis. 

This is a work in progress, and is likely to slow down in the coming months as I finalise my Phd. I've restructured the various templates from the original pipeline to illustrate the EEG data analysis steps with the new code structure. 

## TO-DO
- Restructure the project and maybe turn into a package (DM if you could help us!)

## Published work
`OKTOS_rsEEG_classic_bp.ipynb` - Anijärv et al. 2023. "Spectral Changes of EEG Following a 6-Week Low-Dose Oral Ketamine Treatment in Adults With Major Depressive Disorder and Chronic Suicidality". International Journal of Neuropsychopharmacology, Volume 26, Issue 4, April 2023, Pages 259–267, https://doi.org/10.1093/ijnp/pyad006

## Requirements
The data processing and analysis is tested with Biosemi 32-channel EEG set. As per Toomas' recommendations, create a [conda environment](https://www.anaconda.com/distribution/) with all the dependencies using the environment.yml file in the original repository. However, down below you can see all the required libraries across parts of the pipeline in case you want to use only a specific notebook.

`conda env create -n EEG-pyline -f environment.yml`

If you want to install all the necessary packages separately then these four installs will cover all the packages.

`conda install -c conda-forge mne`
`conda install -c conda-forge autoreject`
`conda install -c nclibz statannotations`
`conda install -c conda-forge fooof`
`conda install -c anaconda openpyxl`

### Pre-processing:
- MNE
- AutoReject

### Spectral analysis + ERP analysis:
- MNE
- specparam (fooof)
- Pandas
- NumPy
- SciPy
- Matplotlib

### Complexity & Entropy analysis
- MNE
- Pandas
- NumPy
- neurokit2

### Data visualisation:
- MNE
- Pandas
- NumPy
- SciPy
- Seaborn
- Matplotlib
- Statannotations

## Citation
If you are using this project in your EEG study, it would be much appreciated if you could cite the origninal repository in your work. See `CITATION.cff` or [![DOI](https://zenodo.org/badge/495300654.svg)](https://zenodo.org/badge/latestdoi/495300654) for information.

If you are using any specific study notebook, then additionally please add citation to the corresponding publication in your article.

## References
[1] Alexandre Gramfort, Martin Luessi, Eric Larson, Denis A. Engemann, Daniel Strohmeier, Christian Brodbeck, Roman Goj, Mainak Jas, Teon Brooks, Lauri Parkkonen, and Matti S. Hämäläinen. MEG and EEG data analysis with MNE-Python. Frontiers in Neuroscience, 7(267):1–13, 2013. doi:10.3389/fnins.2013.00267.

[2] Mainak Jas, Denis Engemann, Federico Raimondo, Yousra Bekhti, and Alexandre Gramfort, “Automated rejection and repair of bad trials in MEG/EEG.” In 6th International Workshop on Pattern Recognition in Neuroimaging (PRNI), 2016.

[3] Mainak Jas, Denis Engemann, Yousra Bekhti, Federico Raimondo, and Alexandre Gramfort. 2017. “Autoreject: Automated artifact rejection for MEG and EEG data”. NeuroImage, 159, 417-429.

[4] McKinney W, others. Data structures for statistical computing in python. In: Proceedings of the 9th Python in Science Conference. 2010. p. 51–6.

[5] Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). DOI: 10.1038/s41586-020-2649-2.

[6] Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, CJ Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E.A. Quintero, Charles R Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, and SciPy 1.0 Contributors. (2020) SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods, 17(3), 261-272.

[7] Waskom M, et al. mwaskom/seaborn: v0.8.1 (September 2017) [Internet]. Zenodo; 2017. Available from: https://doi.org/10.5281/zenodo.883859

[8] J. D. Hunter, "Matplotlib: A 2D Graphics Environment", Computing in Science & Engineering, vol. 9, no. 3, pp. 90-95, 2007.

[9] Donoghue T, Haller M, Peterson EJ, Varma P, Sebastian P, Gao R, Noto T, Lara AH, Wallis JD, Knight RT, Shestyuk A, Voytek B (2020). Parameterizing neural power spectra into periodic and aperiodic components. Nature Neuroscience, 23, 1655-1665. DOI: 10.1038/s41593-020-00744-x

[10] Makowski, D., Pham, T., Lau, Z. J., Brammer, J. C., Lespinasse, F., Pham, H.,
Schölzel, C., & Chen, S. A. (2021). NeuroKit2: A Python toolbox for neurophysiological signal processing.
Behavior Research Methods, 53(4), 1689–1696. https://doi.org/10.3758/s13428-020-01516-y
