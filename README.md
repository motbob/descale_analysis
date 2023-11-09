# descale_analysis

Prerequisites:

- [vapoursynth](https://github.com/vapoursynth/vapoursynth)
- [vs-kernels](https://github.com/Jaded-Encoding-Thaumaturgy/vs-kernels) (pip install vskernels)

This is a tool to determine which scenes in a show are descalable. If a show has multiple kernels, this tool makes it easy to choose how to descale each scene. The tool creates ranges that can be fed to jvsfunc.rfs:

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

Note the `avg_error_thr` and `ind_error_thr` parameters in `get_descale_ranges`. Those are error thresholds that will reject a scene's descalability if they are exceeded. `avg_error_thr` is an error threshold for the scene, and `ind_error_thr` is an error threshold for each individual frame. The default values are *extremely* conservative and should be changed to fit the project.

You can use `test_descale_error` to determine good values for these thresholds:

```py
kernel0 = dict(width=1280, height=720, fractional=719.8, kernel=Bicubic(0, 0.5))
test = test_descale_error(jpn, kernel0)
test.set_output()
```

The error values will appear in the top left of the frame.

As the first code example indicates, you can feed the function a "fractional" number (e.g. from getfnative) or a set of src_\* values. You can't use both, since src_\* values are derived from the `fractional` value.

You can use this tool to detect descalable scenes in single-kernel shows:

```py
kernel = dict(width=1280, height=720, kernel=Bicubic(0, 0.5))

ranges_list = get_descale_ranges(src, [kernel], "showtitle_epnum", avg_error_thr=0.015, ind_error_thr=0.02)

descalable_ranges = ranges_list[0]
```
