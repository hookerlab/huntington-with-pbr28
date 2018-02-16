# Neuroinflammation in Huntington's Disease: new insights with 11C-PBR28 PET/MRI

[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org/repo/hookerlab/huntington-with-pbr28)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1174364.svg)](https://doi.org/10.5281/zenodo.1174364)

This is the companion code and data for the research paper *"Neuroinflammation in Huntington's Disease: new insights with 11C-PBR28 PET/MRI"*, by Cristina Lois, Iv&aacute;n Gonz&aacute;lez, David Izquierdo-Garc&iacute;a, Nicole R. Z&uuml;rcher, Paul Wilkens, Marco L. Loggia, Jacob M. Hooker, and Diana H. Rosas.

## Running the code

You can run the code from your browser by clicking here: [![Binder](http://mybinder.org/badge.svg)](http://mybinder.org/repo/hookerlab/huntington-with-pbr28)

Alternatively you can download the code and run it in your computer as a Jupyter notebook:

1. [Download this repo](https://github.com/hookerlab/huntington-with-pbr28/archive/master.zip) and unzip it.
2. [Install Anaconda for your operating system](https://www.continuum.io/downloads).
3. Create an new `conda` environment and activate it, and run a `jupyter` notebook:
```
$ cd huntington-with-pbr28
$ conda create --name huntington-with-28 -f environment.yml jupyter
$ source activate
$ jupyter notebook
```

## Interactive viewer for the images

You can get an interactive view of the MNI images at [neurovault.org](http://neurovault.org/collections/GHXGLWPB/)

## Citation

This is research work. In case you use any of this code or data, please cite our paper:

```
@unpublished{lois2016,
author={Cristina Lois, Iv\'an Gonz\'alez, David Izquierdo-Garc\'{\i}a, Nicole R. Z\"urcher, Paul Wilkens, Marco L. Loggia, Jacob M. Hooker, and Diana H. Rosas},
title={Neuroinflammation in Huntington's Disease: new insights with [$^{11}$C]-PBR28 PET/MRI},
year={2016},
note={to be published}
}
```

## License

Both the code are data are distributed under open licenses.  The code is distributed under the MIT license (see [LICENSE](LICENSE)). The data (i.e. the files under `data/`) is distributed under the CC0 license (see [LICENSE](./data/LICENSE)).
