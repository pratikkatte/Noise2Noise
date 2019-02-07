## Image Restoration

This is a pytorch implementation of "Noise2Noise: Learning Image Restoration without Clean Data" [1].

## Dependencies
* pytorch
* tensorflow
* numpy

## Train Noise2Noise
### Download Dataset
```
mkdir data
wget http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar
tar -C data -xvf yourfile.tar

```
Any dataset can be used in training and validation instead of the above dataset.

### Train Model
Please see `python3 train.py [-h]` for argument options.
[TODO] create a config file.

### Train with Gaussian noise

[TODO]
### Train with Poissons noise

[TODO]
### Train with Textual noise


### Results

## References
[1] J. Lehtinen, J. Munkberg, J. Hasselgren, S. Laine, T. Karras, M. Aittala, T. Aila, "Noise2Noise: Learning Image Restoration without Clean Data," in Proc. of ICML, 2018.

