# NeuRay

## [Project page](https://liuyuan-pal.github.io/NeuRay/) | [Paper](https://arxiv.org/abs/2107.13421)

## Todo List

- [x] Generalization models and rendering codes.
- [ ] Training of generalization models.
- [ ] Finetuning codes and finetuned models.

## Usage
### Setup
```shell
git clone git@github.com:liuyuan-pal/NeuRay.git
cd NeuRay
pip install -r requirements.txt
```
<details>
  <summary> Dependencies </summary>

  - torch==1.7.1
  - opencv_python==4.4.0
  - tensorflow==2.4.1
  - numpy==1.19.2
  - scipy==1.5.2

</details>

### Download datasets and pretrained models
1. Download processed datasets: [DTU-Test](https://drive.google.com/file/d/1E53MjQJR-cSndJi14MXjx1ombzSq7XTu/view?usp=sharing) / [LLFF](https://drive.google.com/file/d/1CPfdekwKM6zt_skb-IYOruN3Dyzwfhdm/view?usp=sharing) / [NeRF Synthetic](https://drive.google.com/file/d/1EEwDBQY2jUNJpptxOZPG9nkHz96HChed/view?usp=sharing).
2. Download pretrained model [NeuRay-Depth](https://drive.google.com/file/d/1ZIwCnnD8avga8f-p5-Z8m1C_lRgdDaLq/view?usp=sharing) and [NeuRay-CostVolume](https://drive.google.com/file/d/16EVg1ql86FlHwP4JaBhR79dT-gBhDUK2/view?usp=sharing).
3. Organize datasets and models as follows
```
NeuRay
|-- data
    |--model
        |-- neuray_gen_cost_volume
        |-- neuray_gen_depth
    |-- dtu_test
    |-- llff_colmap
    |-- nerf_synthetic
```

### Render
```shell
# render on lego of the NeRF synthetic dataset
python render.py --cfg configs/gen/neuray_gen_depth.yaml \  
                 --database nerf_synthetic/lego/black_800 \ # nerf_synthetic/lego/black_400
                 --pose_type eval                 

# render on snowman of the DTU dataset
python render.py --cfg configs/gen/neuray_gen_depth.yaml \  
                 --database dtu_test/snowman/black_800 \ # dtu_test/snowman/black_400
                 --pose_type eval 
                 
# render on fern of the LLFF dataset
python render.py --cfg configs/gen/neuray_gen_depth.yaml \
                 --database llff_colmap/fern/high \ # llff_colmap/fern/low
                 --pose_type eval
```
The rendered images locate in `data/render/<database_name>/<renderer_name>-pretrain-eval/`.
If the `pose_type` is `eval`, we also generate ground-truth images in `data/render/<database_name>/gt`. 

#### Explanation on parameters of `render.py`.

- `cfg` is the path to the renderer config file, which can also be `configs/gen/neuray_gen_cost_volume.yaml`
- `database` is a database name consisting of `<dataset_name>/<scene_name>/<scene_setting>`.
  -  `nerf_synthetic/lego/black_800` means the scene "lego" from the "nerf_synthetic" dataset using "black" background and the resolution "800$\times$800".
  - `dtu_test/snowman/black_800` means the scene "snowman" from the "dtu_test" dataset using "black" background and the resolution "800$\times$600".
  - `llff_colmap/fern/high` means the scene "fern" from the "llff_colmap" dataset using "high" resolution (1008$\times$756).
  - We may also use `llff_colmlap/fern/low` which renders with "low" resolution (504$\times$378)

### Evaluation

```shell
# psnr/ssim/lpips will be printed on screen
python eval.py --dir_pr data/render/<database_name>/<renderer_name>-pretrain-eval \
			   --dir_gt data/render/<database_name>/gt

# example of evaluation on "fern".
# note we should already render images in the "dir_pr".
python eval.py --dir_pr data/render/llff_colmap/fern/high/neuray_gen_depth-pretrain-eval \
               --dir_gt data/render/llff_colmap/fern/high/gt
```

### Render on custom scenes

To render on custom scenes, please refer to [this](custom_rendering.md)

![](assets/desktop.gif)

## Code explanation

We have provided explanation on variable naming convention in [here](codes_expalanations.md) to make our codes more readable.

## Acknowledgements
In this repository, we have used codes or datasets from the following repositories. 
We thank all the authors for sharing great codes or datasets.

- [IBRNet](https://github.com/googleinterns/IBRNet)
- [MVSNet-official](https://github.com/YoYo000/MVSNet) and [MVSNet-kwea123](https://github.com/kwea123/CasMVSNet_pl)
- [BlendedMVS](https://github.com/YoYo000/BlendedMVS)
- [NeRF-official](https://github.com/bmild/nerf) and [NeRF-torch](https://github.com/yenchenlin/nerf-pytorch)
- [MVSNeRF](https://github.com/apchenstu/mvsnerf)
- [PixelNeRF](https://github.com/sxyu/pixel-nerf)
- [COLMAP](https://github.com/colmap/colmap)
- [IDR](https://lioryariv.github.io/idr/)
- [RealEstate10K](https://google.github.io/realestate10k/)
- [DeepView](https://augmentedperception.github.io/deepview/)
- [Google Scanned Objects](https://app.ignitionrobotics.org/GoogleResearch/fuel/collections/Google%20Scanned%20Objects)
- [LLFF](https://github.com/Fyusion/LLFF)
- [DTU](https://roboimagedata.compute.dtu.dk/?page_id=36)

## Citation
```
@inproceedings{liu2022neuray,
  title={Neural Rays for Occlusion-aware Image-based Rendering},
  author={Liu, Yuan and Peng, Sida and Liu, Lingjie and Wang, Qianqian and Wang, Peng and Theobalt, Christian and Zhou, Xiaowei and Wang, Wenping},
  booktitle={CVPR},
  year={2022}
}
```