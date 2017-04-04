# Improved Dense Trajectories

This library adds some changes to the original [Improved Dense Trajectories  code](http://lear.inrialpes.fr/~wang/improved_trajectories):

- TVL1 flow on the GPU instead of Farneback flow on the CPU.
- Changes for showing and saving images, descriptors etc.

*Copyright (C) 2011 Heng Wang*<br/>
Please cite the original work when using this code:

```
@inproceedings{wang2011action,
  title={Action recognition by dense trajectories},
  author={Wang, Heng and Kl{\"a}ser, Alexander and Schmid, Cordelia and Liu, Cheng-Lin},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2011 IEEE Conference on},
  pages={3169--3176},
  year={2011},
  organization={IEEE}
}
```

## Compiling

In order to compile the improved trajectories code, you need to have the following libraries installed in your system:

- **OpenCV library** (tested with OpenCV-2.4.13, needs to be compiled with `non_free` and `gpu` modules)
- **ffmpeg library** (tested with ffmpeg-2.8.7)

Currently, the libraries are the latest versions. In case they will be out of date, you can also find them on our website: http://lear.inrialpes.fr/people/wang/improved_trajectories
If these libraries are installed correctly, simply type `make` to compile the code. The executable will be in the directory `./release/`.

## Compute features on a Test Video ###

Once you are able to decode the video, computing our features is simple:

`./release/DenseTrackStab ./test_sequences/person01_boxing_d1_uncomp.avi | gzip > out.features.gz`

Now you want to compare your file out.features.gz with the file that we have computed to verify that everything is working correctly. To do so, type:

`vimdiff out.features.gz ./test_sequences/person01_boxing_d1.gz`

Note that due to different versions of codecs, your features may be slightly different with ours. But the major part should be the same.

Due to the randomness of RANSAC, you may get different features for some videos. But for the example "person01_boxing_d1_uncomp.avi", I don't observe any randomness.
There are more explanations about our features on the website, and also a list of FAQ.
