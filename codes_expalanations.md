We have provided some explanation on our codes. Welcome for any questions or discussions.

### Explanations on variable names

- `que` means "query", which is  the "test view" in the paper. e.g. `que_imgs_info` contains information about the test views.
  - `que_imgs_info['Ks']` has the size `qn*3*3` is the intrinsics matrices of the test views. `qn` means the number of test views in this batch. (`qn` means "query number")
  - `que_imgs_info['poses']` has the size `qn*3*4` is the pose matrices $[R;t]$ of the test views. We use the opencv-style poses which converts the scene coordinate to the camera coordinate $x_{cam}= Rx_{scene}+t$.  
  - `que_imgs_info['depth_range']` has the size `qn*2`, which are the near plane depth, and the far plane depth.
  - `que_imgs_info['coords']` has the size `qn*rn*2`, which are `rn` 2D coordinates in pixel on test views. We will render the rays emitted from these coordinates. (`rn` means "ray number")
  - `que_depth` has the size `qn*rn*dn` and is the sample depth values on test rays. `dn` means the number of points sampled on a test ray. ($K_t$ in the paper)
- `ref` means "reference", which is the "input view" in the paper. e.g. `ref_imgs_info` contains information about the input views.
  - `ref_imgs_info['ray_feats']` has the size `rfn*f*h*w`, which is the visibility feature map $G$ on input views. `rfn` means the "reference view number", i.e. the number of input views (working view number $N_w$ in the paper). `f` means the dimension number. `h*w` is the size of this feature map.
- `prj` means information about projected sample points on input views. e.g. `prj_dict`
- `nr` means "network rendering", which is computed from the constructed radiance fields. e.g. `pixel_colors_nr` means the output colors computed by volume rendering on the constructed radiance field.
- `dr` means "direction rendering", which is directly computed from the NeuRay representation.
- Summary of matrix size in annotations
  - `qn` test view number
  - `rn` test ray number
  - `rfn` input working view number $N_w$
  - `dn` sample point number on a test ray $K_t$
  - `f` feature dimension
  - `pn=qn*rn*dn` total sample point number
  
### Dataset management

All datasets are managed by `BaseDatabase` in `dataset/database.py`. 
If we want to extend to a new dataset, we can write a new subclass of `BaseDatabase` and implement all its functions.