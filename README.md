# descale_analysis

Prerequisites:

- [vapoursynth](https://github.com/vapoursynth/vapoursynth)
- [vs-jetpack](https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack)
- [mvsfunc](https://github.com/HomeOfVapourSynthEvolution/mvsfunc)

This is a tool to determine which scenes in a show are descalable. If a show has multiple kernels, this tool makes it easy to choose how to descale each scene. The tool creates ranges that can be fed to vstools.replace_ranges:

```py
from descale_analysis import test_descale_error, get_descale_ranges
from vskernels import Bilinear, Bicubic, Lanczos

kernelres0 = dict(width=1280, height=720, kernel=Bicubic(0, 0.5))
kernelres1 = dict(width=1280, height=720, fractional=719.8, kernel=Bicubic(0, 1))
kernelres2 = dict(width=1280, height=720, kernel=Bilinear(), src_top=0.2, src_height=719.6, src_left=0.2, src_width=1279.6)

ranges_list = get_descale_ranges(src, [kernelres0, kernelres1, kernelres2], "showtitle_epnum", avg_error_thr=0.015, ind_error_thr=0.02)

kernelres0_descalable_ranges = ranges_list[0]
kernelres1_descalable_ranges = ranges_list[1]
kernelres2_descalable_ranges = ranges_list[2]
```

The ranges will also get written to text files so that you don't have to run the analysis again if the encode aborts for whatever reason. The names of the text files are determined by `txtfilename`. If you don't want to write to text files, you can set `txtfilename=None`.

Note the `avg_error_thr` and `ind_error_thr` parameters in `get_descale_ranges`. Those are error thresholds that will reject a scene's descalability if they are exceeded. `avg_error_thr` is an error threshold for the scene, and `ind_error_thr` is an error threshold for each individual frame. The default values are *extremely* conservative and should be changed to fit the project.

You can pass additional arguments to the descaler, like blur, with scale_args. E.g., `kernelres0 = dict(width=1280, height=720, kernel=Bicubic(0,0.5), scale_args=dict(blur=0.9))`

The tool will pick the descale kernel/res with the lowest error if more than one of them fall under the error thresholds. It won't return multiple possible kernels for the same scene.

You can use `test_descale_error` to determine good values for these thresholds:

```py
kernelres0 = dict(width=1280, height=720, fractional=719.8, kernel=Bicubic(0, 0.5))
test = test_descale_error(jpn, kernelres0)
test.set_output()
```

The error values will appear in the top left of the frame.

As the first code example indicates, you can feed the function a "fractional" number (e.g. from getfnative) or a set of src_\* values. You can't use both, since src_\* values are derived from the `fractional` value.

You can use this tool to detect descalable scenes in single-kernel shows:

```py
kernelres = dict(width=1280, height=720, kernel=Bicubic(0, 0.5))

ranges_list = get_descale_ranges(src, [kernelres], "showtitle_epnum", avg_error_thr=0.015, ind_error_thr=0.02)

descalable_ranges = ranges_list[0]
```

### The "fake kernel" method of scene checking

Sometimes, you will be faced with a show where error thresholds don't work all that well. For example, there may be a show where, due to clipping or compression artifacts, many scenes have high error but still benefit from a descale.

In cases like that, you may want to set a high error threshold and use a fake kernel in order to exclude scenes that do not appear to be descalable.

Suppose you have a show that appears to be 1280x720 bilinear. You can run the following code:

```py
kernel0 = dict(width=1280, height=720, kernel=Bilinear())
kernel1 = dict(width=1344, height=756, kernel=Bilinear())#fake kernel

get_descale_ranges(src, [kernel0, kernel1], "showtitle_epnum", avg_error_thr=0.03, ind_error_thr=0.05)
```

Because of the way descale_analysis calculates error (MSE), if a scene has lower error for kernel1 than kernel0, then it's unlikely that that scene is actually 720p bilinear, even if its error is low on an absolute basis. The above code will prevent those kinds of scenes from being included in the kernel0 ranges.

If you use this method, you will usually find that very blurry scenes are placed in the fake kernel ranges (since the information in the image is dominated by random noise at that point). Most people edgemask their rescales anyway, and so this is normally no great loss.

### Various other tools for checking descales

There are various other functions that allow you to get more information about descales. For example:

`checkboth`: input a single frame and res information (e.g. `dict(width=1280, height=720)`; supports fractional and raw src_ values), output a set of error maps that are labeled in the top left.  
`checkbothextended`: same as above except that it returns more error maps.