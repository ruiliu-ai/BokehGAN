Pre-trained model
======
[Click here to download](https://drive.google.com/file/d/1cMf8ThVeKT28cIUjexnyfxiI-pauOe0p/view?usp=sharing)

Save the pre-trained models to checkpoints/bokeh_saliency/

Prepare coarse masks
======
You can directly download them [here](https://drive.google.com/file/d/1Cz-ICK2caxhrrApcLHjY6EYleqR0C87v/view?usp=sharing).

Or you can follow the [open-source saliency detection methods](https://github.com/Joker316701882/Salient-Object-Detection) to obtain the masks on your own.

Then you should save them to [path to masks]

Synthesize images
======
python generate.py --name bokeh_saliency --which_epoch 350 --dataset_mode custom --origin_dir [path to original images] --masks_dir [path to masks]
