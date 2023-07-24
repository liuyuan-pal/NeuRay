### Render with custom scenes

We have provided interfaces to render on custom scenes. An example is in the following.

#### 1. Download example images 

Download images [here](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/yuanly_connect_hku_hk/EVGakxQWw5ZLjgK3147QCNQBpZu5U75mTU9-I-vkkAfyvA?e=gskW7j) and organize files like

```
NeuRay
|-- data
    |-- example
        |-- desktop
            |-- images
                |-- *.jpg
```

Make sure these images are not too large (e.g. <= 960x720). Otherwise, running COLMAP or rendering will take very long time.

#### 2. Run COLMAP to reconstruct camera poses and scene geometry

```shell
python run_colmap.py --example_name desktop \
                     --colmap <path-to-your-colmap> # note we need the dense reconstruction
```

`data/example/desktop` will be the project directory for COLMAP.

#### 3. Render on the scene

```shell
python render.py --database example/desktop/raw --cfg configs/gen/neuray_gen_depth.yaml --pose_type circle
```

This command will render images using a set of circle poses. Renderings are saved in `data/render/example/desktop/raw/neuray_gen_depth-pretrain-circle`.

Optionally, we can also render low resolution images with

```
python render.py --database example/desktop/480 --cfg configs/gen/neuray_gen_depth.yaml --pose_type circle
```

#### 4. Make video

```shell
ffmpeg.exe -y -framerate 30 -r 30 -i data/render/example/desktop/raw/neuray_gen_depth-pretrain-circle/%d-nr_fine.jpg -vcodec libx264 -pix_fmt yuv420p -filter:v "crop=600:800:60:80" desktop.mp4
```

A video looking like the following should be generated.

 ![](assets/desktop.gif)

### Render on BlendedMVS dataset

We have provided estimated depth maps from COLMAP on the BlendedMVS dataset, which can be downloaded [here](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/yuanly_connect_hku_hk/EaUz_5npniRMhoiZg1U6uBYBqHdBIcABhL2z9UQjgvVKyg?e=g7Kxig), and the corresponding rendering poses are in `configs/inter_trajectory/blended_mvs`. Organize files like `data/blended-mvs/<uids>` and we can render novel view images by

```shell
# We may change the "building" in the command to ["iron_dog", "santa", "dragon", "mermaid", "laid_man"]
python render.py --cfg configs/gen/neuray_gen_depth.yaml \
                 --database blended_mvs/building/half \ 
                 --pose_type inter_60 \
                 --pose_fn configs/blended_mvs/building.txt
```

 Rendered images are saved in `data/render/blended_mvs/building/half/neuray_gen_depth-pretrain-inter_60`.
 
### Finetuning on a custom scene

```shell
python run_training.py --cfg configs/train/ft/example/neuray_ft_desktop_pretrain.yaml
```
To write a new config file for finetuning on a new custom scene. You need to change all `database_name` and `database_split` in the `.yaml` file.
The `database_name` will be parsed to a `BaseDatabase` class in `dataset/database.py` by the function `parse_database_name`, which are utilized by NeuRay as explained in [codes_explanations.md](codes_explanations.md).
The `database_split` will be used in the function `get_database_split` in the `dataset/database.py` to determine training images and validation images.
