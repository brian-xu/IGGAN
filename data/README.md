# Model Voxelization

Generates many binvox files for use in pretraining the neural renderer.

The voxelization loop is written for Windows. Other systems will need to download the correct voxelizer from
[Patrick Min](https://www.patrickmin.com/binvox/) and make modifications to this line.

```33| os.system(f"binvox.exe -d 64 -cb {objs[i]}")```

In addition, the `binvox.exe` check is Windows-specific and can just be deleted.

## Notes

Some models will lose a lot of their geometry during the resizing process (-d 64). I have three things to say about
this:

1. The architecture is fully convolutional, so you can just double the voxel grid size to capture more detail
   (Note that you will also have to double the renderer output size). However, this will octuple the complexity due to
   the square-cube law. That's not a problem if you use the patch training method described in the
   [original RenderNet paper](https://papers.nips.cc/paper/2018/file/68d3743587f71fbaa5062152985aff40-Paper.pdf), but
   this isn't a full implementation of that code and therefore doesn't implement it.
2. The neural renderer and loss criteria are given the same voxel input, so the lost information really isn't a problem
   anyways. It will learn to correctly associate the missing voxels with empty space.
3. This is only intended as pretraining data. The renderer will be fine-tuned during the main training loop to render
   the target category of objects. Pretraining the neural renderer speeds up convergence of the main GAN.

## Credits

Models downloaded from [ShapeNet](https://www.shapenet.org/).

Voxelizer from [Patrick Min](https://www.patrickmin.com/binvox/).