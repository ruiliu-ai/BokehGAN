BokehGAN
======
This is one of the approaches participating in AIM 2019 challenge on bokeh effect synthesis. For more methods and results, please refer to the [summary](https://ieeexplore.ieee.org/abstract/document/9022578/). 

Pre-trained model
======
[Click here to download](https://drive.google.com/file/d/1cMf8ThVeKT28cIUjexnyfxiI-pauOe0p/view?usp=sharing)

Save the pre-trained models to checkpoints/bokeh_saliency/

Prepare coarse masks
======
You can directly download them [here](https://drive.google.com/file/d/1gYsPfDsLQEIeW9ihza2c0wqf7L6-hs8t/view?usp=sharing).

Or you can follow the [open-source saliency detection methods](https://github.com/Joker316701882/Salient-Object-Detection) to obtain the masks on your own.

Then you should save them to [path to masks]

Synthesize images
======
python generate.py --name bokeh_saliency --which_epoch 350 --dataset_mode custom --origin_dir [path to original images] --masks_dir [path to masks]

Citation
======
If you find this project useful in your research, please consider citing:
```
@INPROCEEDINGS{9022578,
  author={A. {Ignatov} and J. {Patel} and R. {Timofte} and B. {Zheng} and X. {Ye} and L. {Huang} and X. {Tian} and S. {Dutta} and K. {Purohit} and P. {Kandula} and M. {Suin} and A. N. {Rajagopalan} and Z. {Xiong} and J. {Huang} and G. {Dong} and M. {Yao} and D. {Liu} and M. {Hong} and W. {Lin} and Y. {Qu} and J. {Choi} and W. {Park} and M. {Kim} and R. {Liu} and X. {Mao} and C. {Yang} and Q. {Yan} and W. {Sun} and J. {Fang} and M. {Shang} and F. {Gao} and S. {Ghosh} and P. K. {Sharma} and A. {Sur} and W. {Yang}},
  booktitle={2019 IEEE/CVF International Conference on Computer Vision Workshop (ICCVW)}, 
  title={AIM 2019 Challenge on Bokeh Effect Synthesis: Methods and Results}, 
  year={2019},
  volume={},
  number={},
  pages={3591-3598},
  doi={10.1109/ICCVW.2019.00444}}
```
