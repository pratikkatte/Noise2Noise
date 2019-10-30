# Im# Noise2Noise: Learning Image Restoration without Clean Data
age Restoration

This is an unofficial PyTorch implementation of [Noise2Noise](https://arxiv.org/abs/1803.04189) (Lehtinen et al. 2018).

## Dependencies

* PyTorch
* Torchvision
* NumPy
* Matplotlib
* Pillow

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

### Gaussian noise
The noise parameter is the maximum standard deviation σ.
```
python3 train.py \
  --train-dir ../data/train --train-size 1000 \
  --valid-dir ../data/valid --valid-size 200 \
  --ckpt-save-path ../ckpts \
  --nb-epochs 10 \
  --batch-size 4 \
  --loss l2 \
  --noise-type gaussian \
  --noise-param 50 \
  --crop-size 64 \
  --plot-stats \
  --cuda
```


## Results

Gaussian model was trained for 100 epochs with a train/valid split of 2000/400.


<table align="center">
  <tr align="center">
    <th colspan=9>Gaussian noise (σ = 25)</td>
  </tr>
  <tr align="center">
    <td colspan=2>Noisy input (20.34 dB)</td>
    <td colspan=2>Denoised (32.68 dB)</td>
    <td colspan=2>Clean targets (32.49 dB)</td>
    <td colspan=2>Ground truth</td>
  </tr>
  <tr align="center">
    <td colspan=2><img src="figures/monarch-gaussian-noisy.png"></td>
    <td colspan=2><img src="figures/monarch-gaussian-denoised.png"></td>
    <td colspan=2><img src="figures/monarch-gaussian-clean.png"></td>
    <td colspan=2><img src="figures/monarch.png"></td>
  </tr> 
</table>

## References
[1] J. Lehtinen, J. Munkberg, J. Hasselgren, S. Laine, T. Karras, M. Aittala, T. Aila, "Noise2Noise: Learning Image Restoration without Clean Data," in Proc. of ICML, 2018.

