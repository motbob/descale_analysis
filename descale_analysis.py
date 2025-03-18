import vapoursynth as vs
from vapoursynth import core
import functools
from vskernels import Kernel, Bicubic, Lanczos, Bilinear, Point
from mvsfunc import ShowAverage

def instantiate_kernel(krnl):
    if isinstance(krnl, type) and (krnl is Kernel or issubclass(krnl, Kernel)):
        krnl = krnl()
    return krnl

def get_cropped_width_height(clip, src_height, src_width, base_height, base_width):
    from math import floor
    assert base_height >= src_height
    cropped_width = base_width - 2 * floor((base_width - src_width) / 2)
    cropped_height = base_height - 2 * floor((base_height - src_height) / 2)
    return cropped_width, cropped_height

def descale_cropping_args(clip: vs.VideoNode, src_height: float, base_height: int, base_width: int, mode: str = 'wh'):
    src_width = src_height * clip.width / clip.height
    cropped_width, cropped_height = get_cropped_width_height(clip, src_height, src_width, base_height, base_width)
    args = dict(
        width = clip.width,
        height = clip.height
    )
    args_w = dict(
        width = cropped_width,
        src_width = src_width,
        src_left = (cropped_width - src_width) / 2
    )
    args_h = dict(
        height = cropped_height,
        src_height = src_height,
        src_top = (cropped_height - src_height) / 2
    )
    if 'w' in mode.lower():
        args.update(args_w)
    if 'h' in mode.lower():
        args.update(args_h)
    return args

def add_frame_average(n, clip, f):
    diff_raw = f.props['PSAverage']
    return core.text.Text(clip, str(diff_raw))

def gen_descale_error(clip: vs.VideoNode, width: float, height: float, kernel: Kernel, src_height: float, src_top: float, src_width: float, src_left: float, thr: float = 0.01, write_error = False, clamp = True) -> vs.VideoNode:
    clip = clip.resize.Point(format=vs.GRAYS, matrix_s='709' if clip.format.color_family == vs.RGB else None)
    descale = kernel.descale(clip, width=width, height=height, src_height=src_height, src_top=src_top, src_width=src_width, src_left=src_left)
    rescale = kernel.scale(descale, width=clip.width, height=clip.height, src_height=src_height, src_top=src_top, src_width=src_width, src_left=src_left)
    if clamp:
        rescale = core.std.Expr([rescale], f'x 1 > 1 x 0 < 0 x ? ?')
        clip = core.std.Expr([clip], f'x 1 > 1 x 0 < 0 x ? ?')
    diff = core.std.Expr([clip, rescale], f'x y - abs dup {thr} > swap 0 ?').std.Crop(10, 10, 10, 10)
    diff = core.std.Expr([diff], f'x 32 *')
    diff = core.std.PlaneStats(diff, prop="PS")
    if write_error:
        diff = core.std.FrameEval(diff, functools.partial(add_frame_average, clip=diff), prop_src=[diff])
    return diff

def blur_bicubic(x, b, c, blur):
    import math
    def normal_bicubic(x):
        if abs(x) < 1:
            return ((12 - 9 * b - 6 * c) * abs(x)**3 + (-18 + 12 * b + 6 * c) * abs(x)**2 + (6 - 2 * b)) / 6
        elif abs(x) < 2:
            return ((-1 * b - 6 * c) * abs(x) ** 3 + (6 * b + 30 * c) * abs(x) ** 2 + (-12 * b - 48 * c) * abs(x) + (8 * b + 24 * c)) / 6
        else:
            return 0
    def blurred(x):
        return normal_bicubic(x/blur)
    if (blurred(x-2) + blurred(x-1) + blurred(x) + blurred(x + 1) + blurred(x + 2)) == 0:
        return 0
    else:
        return blurred(x) / (blurred(x-2) + blurred(x-1) + blurred(x) + blurred(x + 1) + blurred(x + 2))

def blur_lanczos(x, taps, blur):
    import math
    def sinc(x):
        return 1.0 if x == 0 else math.sin(x * math.pi) / (x * math.pi)
    return sinc(x/blur) * sinc(x/blur / taps)

#experimental
def gen_descale_error_blur(clip: vs.VideoNode, width, height, src_height, src_top, src_width, src_left, b, c, blur, thr = 0.01, write_error = False, taps=None, bilinear=False) -> vs.VideoNode:
    clip = clip.resize.Point(format=vs.GRAYS, matrix_s='709' if clip.format.color_family == vs.RGB else None)
    if bilinear == True:
        descale = core.descale.Debilinear(clip, width=width, height=height, src_top = src_top, src_height = src_height, src_width=src_width, src_left=src_left, blur=blur)
        rescale = core.fmtc.resample(descale, w = clip.width, h = clip.height, kernel = "bilinear", fh = 1/blur, fv = 1/blur, sx=src_left, sy=src_top, sw=src_width, sh=src_height)
    elif taps:
        descale = core.descale.Delanczos(clip, width=width, height=height, src_top = src_top, src_height = src_height, src_width=src_width, src_left=src_left, blur=blur, taps=taps)
        rescale = core.fmtc.resample(descale, w = clip.width, h = clip.height, kernel = "lanczos", a1 = taps, fh = 1/blur, fv = 1/blur, sx=src_left, sy=src_top, sw=src_width, sh=src_height)
    else:
        descale = core.descale.Debicubic(clip, width=width, height=height, src_top = src_top, src_height = src_height, src_width=src_width, src_left=src_left, blur=blur, b=b, c=c)
        rescale = core.fmtc.resample(descale, w = clip.width, h = clip.height, kernel = "bicubic", a1 = b, a2 = c, fh = 1/blur, fv = 1/blur, sx=src_left, sy=src_top, sw=src_width, sh=src_height)
    diff = core.std.Expr([clip, rescale], f'x y - abs dup {thr} > swap 0 ?').std.Crop(10, 10, 10, 10)
    diff = core.std.Expr([diff], f'x 32 *')
    if write_error:
        diff = core.std.PlaneStats(diff, prop="PS")
        diff = core.std.FrameEval(diff, functools.partial(add_frame_average, clip=diff), prop_src=[diff])
    return diff

def add_mask_value(clip):
    comp_mask = clip.resize.Point(format=vs.GRAYS, matrix_s='709' if clip.format.color_family == vs.RGB else None)
    comp_mask = core.std.Sobel(comp_mask)
    comp_mask = core.std.PlaneStats(comp_mask, prop='Mask')
    clip_wtf = core.std.CopyFrameProps(clip, comp_mask)
    return clip_wtf

def process_descale_settings_dict(clip, descale_settings, res_only=False):
    if "width" not in descale_settings or "height" not in descale_settings:
        raise Exception("You need to set both height and width!")
    width = descale_settings["width"]
    height = descale_settings["height"]
    if not isinstance(height, int):
        raise Exception("height must be an int")
    if not isinstance(width, int):
        raise Exception("width must be an int")
    if "blur" in descale_settings:
        if "src_height" in descale_settings or "src_width" in descale_settings or "src_left" in descale_settings or "src_top" in descale_settings or "fractional" in descale_settings:
            raise Exception("Blur only works with pure fractional descales at the moment.")
        blur = descale_settings["blur"]
    else:
        blur = None
    if "fractional" in descale_settings:
        descale_type = "fractional"
        if "src_height" in descale_settings or "src_width" in descale_settings or "src_left" in descale_settings or "src_top" in descale_settings:
            raise Exception("You can't set both fractional and src_ values")
        cropping_args = descale_cropping_args(clip, src_height=descale_settings["fractional"], base_height=height, base_width=width)
        src_height, src_width, src_top, src_left = cropping_args["src_height"], cropping_args["src_width"], cropping_args["src_top"], cropping_args["src_left"]
    else:
        src_height = descale_settings.get("src_height", descale_settings["height"])
        src_width = descale_settings.get("src_width", descale_settings["width"])
        src_top = descale_settings.get("src_top", 0)
        src_left = descale_settings.get("src_left", 0)
        if src_height == descale_settings["height"] and src_width == descale_settings["width"] and src_top == 0 and src_left == 0:
            descale_type = "integer"
        else:
            descale_type = "manual"
    processed_dict = dict(width=width, height=height, src_height=src_height, src_width=src_width, src_top=src_top, src_left=src_left, descale_type=descale_type, blur=blur)
    if not res_only:
        kernel = instantiate_kernel(descale_settings["kernel"])
        if not isinstance(kernel, Kernel):
            raise Exception("'kernel' needs to be a vskernels kernel, e.g. vskernels.Bilinear() etc.")
        processed_dict["kernel"] = kernel
    return processed_dict

def test_descale_error(clip, descale_settings, thr=0.01, clamp=True):
    clip = clip.resize.Point(format=vs.GRAYS)
    clip = add_mask_value(clip)
    from vskernels import Kernel
    #descale_settings needs to be a dict with width, height, "kernel" (a vs-kernels object),
    #and optionally EITHER src_ values or "fractional", a float.
    #src_ values are the descale versions. "fractional" is equivalent to getfnative output.
    descale_settings = process_descale_settings_dict(clip, descale_settings)
    width, height, kernel, src_top, src_height, src_width, src_left = descale_settings["width"], descale_settings["height"], instantiate_kernel(descale_settings["kernel"]), descale_settings["src_top"], descale_settings["src_height"], descale_settings["src_width"], descale_settings["src_left"]
    def get_calc(n, f, clip, core):
        diff_raw = f.props['PSAverage']
        mask_value = f.props['MaskAverage']
        if mask_value == 0:
            diff_primary = 0
        else:
            diff_primary = diff_raw / mask_value
        return core.text.Text(clip, str(diff_primary))
    diff = gen_descale_error(clip, width=width, height=height, kernel=kernel, src_top = src_top, src_height = src_height, src_width=src_width, src_left=src_left, thr=thr, clamp=clamp)
    diff = core.std.PlaneStats(diff, prop='PS')
    return core.std.FrameEval(diff, functools.partial(get_calc, clip=diff, core=vs.core), prop_src=[diff])

def get_descale_ranges(clip, kernels, txtfilename=None, ind_error_thr = 0.01, avg_error_thr = 0.006, thr = 0.01, dfttest=False):
    if isinstance(kernels, dict):
        kernels = [kernels]
    clipdown = core.resize.Bicubic(clip, 854, 480, format=vs.YUV420P8)
    clipdown = core.wwxd.WWXD(clipdown)
    clip = core.resize.Point(clip, format=vs.GRAYS)
    blur=False
    for kernel in kernels:
        descale_settings = process_descale_settings_dict(clip, kernel)
        if descale_settings["blur"]:
            blur = True
    if dfttest:
        from vsdenoise import DFTTest
        clip = DFTTest.denoise(clip)
    clip = add_mask_value(clip)
    kernel_appends = []
    kerneldiffs = []
    frame_ranges = []
    defective = []
    total_error = []
    start = []
    end = []
    notcatalogued = []
    biases = []
    good = []
    for kernel in kernels:
        descale_settings = process_descale_settings_dict(clip, kernel)
        bias = kernel.get("bias", 1)
        biases.append(bias)
        #ind_error_ker = kernel[8]
        #avg_error_ker = kernel[9]
        descale_kernel = instantiate_kernel(kernel["kernel"])
        if hasattr(descale_kernel, "b"):
            kernelappend = f"bicubic_{getattr(descale_kernel, 'b')}_{getattr(descale_kernel, 'c')}"
        elif hasattr(descale_kernel, "taps"):
            kernelappend = f"lanczos{getattr(descale_kernel, 'taps')}"
        else:
            kernelappend = f"bilinear"
        if descale_settings["descale_type"] == "integer":
            kernelappend += f"_{descale_settings['width']}_{descale_settings['height']}"
        elif descale_settings["descale_type"] == "fractional":
            kernelappend += f"_{kernel['fractional']}"
        else:
            kernelappend += f"_{descale_settings['width']}_{descale_settings['height']}_{descale_settings['src_top']}_{descale_settings['src_height']}_{descale_settings['src_left']}_{descale_settings['src_width']}"
        kernel_appends.append(kernelappend)
        if blur:
            print("shouldn't be seeing this message unless you're doing blur")
            kernelres = descale_settings
            diff = gen_descale_error_blur(clip, kernelres=kernelres, thr=thr)
        else:
            diff = gen_descale_error(clip, kernel=descale_settings['kernel'], width=descale_settings['width'], height=descale_settings['height'], src_top=descale_settings['src_top'], src_height=descale_settings['src_height'], src_width=descale_settings['src_width'], src_left=descale_settings['src_left'], thr=thr)
        diff = core.std.PlaneStats(diff, prop='PS')
        kerneldiffs.append(diff)
        frame_ranges.append([])
        defective.append(0)
        total_error.append(0)
        start.append(0)
        end.append(0)
        notcatalogued.append(False)
        good.append(False)
    nokernel_ranges = []
    nokernel_start = 0
    nokernel_end = 0
    nokernel_good = False
    
    frames = 0
    for n in range(len(clip)):
        if (clipdown.get_frame(n).props.Scenechange == 1) and (n != 0):
            avg_error = []
            #calculate avg error for each kernel, applying bias as necessary
            for m in range(len(total_error)):
                bias = biases[m]
                average = total_error[m]/frames/bias
                #if average < 0.000005:
                #    average = 0
                avg_error.append(average)
            #if multiple kernels are returning the same low error,
            #we shouldn't be rescaling the scene
            lowest_error = min(avg_error)
            #check if all kernels are defective so we can add it to the no-kernel list.
            #we don't add "all kernels were OK" scenes to that list.
            #we want the list to contain interesting scenes that may need
            #special handling
            all_defective = True
            matches_lowest = 0
            for m in range(len(total_error)):
                # if kernels[m][9] != None:
                #     avg_error_temp = kernels[m][9]
                # else:
                #     avg_error_temp = avg_error_thr
                avg_error_temp = avg_error_thr
                if avg_error[m] != lowest_error or avg_error[m] > avg_error_temp:
                    defective[m] = 1
                if avg_error[m] == lowest_error:
                    matches_lowest += 1
                if defective[m] == 0:
                    all_defective = False
            if matches_lowest > 1:
                for m in range(len(total_error)):
                    defective[m] = 1
            for m in range(len(total_error)):
                total_error[m] = 0
                if defective[m] == 0:
                    end[m] = n - 1
                    good[m] = True
                else:
                    if good[m] == True:
                        frame_ranges[m].append((start[m], end[m]))
                        print(f"{kernel_appends[m]} is {frame_ranges[m]}")
                    good[m] = False
                    start[m] = n
                    defective[m] = 0
            if all_defective:
                nokernel_end = n - 1
                nokernel_good = True
            else:
                if nokernel_good == True:
                    nokernel_ranges.append((nokernel_start, nokernel_end))
                    print(f"no-kernel is {nokernel_ranges}")
                nokernel_good = False
                nokernel_start = n
            frames = 0
        if (n == len(clip) - 1):
            avg_error = []
            for m in range(len(total_error)):
                bias = biases[m]
                if bias == None:
                    bias = 1
                avg_error.append(total_error[m]/frames/bias)
            lowest_error = min(avg_error)
            for m in range(len(total_error)):
                # if kernels[m][9] != None:
                #     avg_error_temp = kernels[m][9]
                # else:
                #     avg_error_temp = avg_error_thr
                avg_error_temp = avg_error_thr
                if avg_error[m] != lowest_error or avg_error[m] > avg_error_temp:
                    defective[m] = 1
                if defective[m] == 1:
                    if good[m] == True:
                        frame_ranges[m].append((start[m], end[m]))
                else:
                    end[m] = n
                    frame_ranges[m].append((start[m], end[m]))
        skip = True
        for m in range(len(total_error)):
            if defective[m] == 0:
                skip = False
        if skip == True:
            frames += 1
            continue
        mask_value = clip.get_frame(n).props.MaskAverage
        for m in range(len(total_error)):
            diff_value = kerneldiffs[m].get_frame(n).props.PSAverage
            if mask_value == 0:
                diff_primary = 0
            else:
                diff_primary = diff_value / mask_value
            total_error[m] = total_error[m] + diff_primary
            # if kernels[m][8] != None:
            #     ind_error_thr_temp = kernels[m][8]
            # else:
            #     ind_error_thr_temp = ind_error_thr
            ind_error_thr_temp = ind_error_thr
            if diff_primary > ind_error_thr_temp:
                defective[m] = 1
        frames += 1
    if txtfilename:
        for m in range(len(total_error)):
            with open(f"{txtfilename}_{kernel_appends[m]}.txt", "w") as x:
                x.write(str(frame_ranges[m]))
        with open(f"{txtfilename}_nokernel.txt", "w") as x:
            x.write(str(nokernel_ranges))
    return frame_ranges

def checkboth(frame, res):
    res_info = process_descale_settings_dict(frame, res, res_only=True)
    blank_clip = core.std.BlankClip(frame, height = frame.height - 20, width = frame.width - 20)
    for i in range(0,4):
        for j in range(0,7):
            b = i
            c = j
            b = b * 0.16666
            c = c * 0.16666
            a = gen_descale_error(frame, res_info["width"], res_info["height"], Bicubic(b, c), res_info["src_height"], res_info["src_top"], res_info["src_width"], res_info["src_left"], write_error=True)
            blank_clip += a
    blank_clip += gen_descale_error(frame, res_info["width"], res_info["height"], Bilinear(), res_info["src_height"], res_info["src_top"], res_info["src_width"], res_info["src_left"], write_error=True)
    blank_clip += gen_descale_error(frame, res_info["width"], res_info["height"], Lanczos(2), res_info["src_height"], res_info["src_top"], res_info["src_width"], res_info["src_left"], write_error=True)
    blank_clip += gen_descale_error(frame, res_info["width"], res_info["height"], Lanczos(3), res_info["src_height"], res_info["src_top"], res_info["src_width"], res_info["src_left"], write_error=True)
    blank_clip += gen_descale_error(frame, res_info["width"], res_info["height"], Lanczos(4), res_info["src_height"], res_info["src_top"], res_info["src_width"], res_info["src_left"], write_error=True)
    blank_clip += gen_descale_error(frame, res_info["width"], res_info["height"], Lanczos(5), res_info["src_height"], res_info["src_top"], res_info["src_width"], res_info["src_left"], write_error=True)
    return blank_clip[1:]

def checkbothextended(frame, res_info):
    res_info = process_descale_settings_dict(frame, res_info, res_only=True)
    blank_clip = core.std.BlankClip(frame, height = frame.height - 20, width = frame.width - 20)
    for i in range(0,4):
        for j in range(0,21):
            b = i
            c = j
            b = b * 1/6
            c = c * 0.05
            a = gen_descale_error(frame, res_info["width"], res_info["height"], Bicubic(b, c), res_info["src_height"], res_info["src_top"], res_info["src_width"], res_info["src_left"], write_error=True)
            blank_clip += a
    blank_clip += gen_descale_error(frame, res_info["width"], res_info["height"], Bilinear(), res_info["src_height"], res_info["src_top"], res_info["src_width"], res_info["src_left"], write_error=True)
    blank_clip += gen_descale_error(frame, res_info["width"], res_info["height"], Lanczos(2), res_info["src_height"], res_info["src_top"], res_info["src_width"], res_info["src_left"], write_error=True)
    blank_clip += gen_descale_error(frame, res_info["width"], res_info["height"], Lanczos(3), res_info["src_height"], res_info["src_top"], res_info["src_width"], res_info["src_left"], write_error=True)
    blank_clip += gen_descale_error(frame, res_info["width"], res_info["height"], Lanczos(4), res_info["src_height"], res_info["src_top"], res_info["src_width"], res_info["src_left"], write_error=True)
    blank_clip += gen_descale_error(frame, res_info["width"], res_info["height"], Lanczos(5), res_info["src_height"], res_info["src_top"], res_info["src_width"], res_info["src_left"], write_error=True)
    return blank_clip[1:]

def checkboth_neg(frame, res):
    res_info = process_descale_settings_dict(frame, res, res_only=True)
    blank_clip = core.std.BlankClip(frame, height = frame.height - 20, width = frame.width - 20)
    for i in range(1,7):
        for j in range(0,7):
            b = -i
            c = j
            b = b * 0.16666
            c = c * 0.16666
            a = gen_descale_error(frame, res_info["width"], res_info["height"], Bicubic(b, c), res_info["src_height"], res_info["src_top"], res_info["src_width"], res_info["src_left"], write_error=True)
            blank_clip += a
    return blank_clip[1:]

def manual_check(frame, res_info, kernels):
    res_info = process_descale_settings_dict(frame, res_info, res_only=True)
    descale = kernels[0].descale(frame, width=res_info["width"], height=res_info["height"], src_left=res_info["src_left"], src_width=res_info["src_width"], src_top=res_info["src_top"], src_height=res_info["src_height"])
    for descale_kernel in kernels:
        descale += descale_kernel.descale(frame, width=res_info["width"], height=res_info["height"], src_left=res_info["src_left"], src_width=res_info["src_width"], src_top=res_info["src_top"], src_height=res_info["src_height"])
    fullres_srctop = res_info["src_top"] * -1 * frame.height / res_info["height"]
    fullres_srcheight = res_info["src_height"] * frame.height / res_info["height"]
    fullres_srcleft = res_info["src_left"] * -1 * frame.width / res_info["width"]
    fullres_srcwidth = res_info["src_width"] * frame.width / res_info["width"]
    descale += core.resize.Bicubic(frame, res_info["width"], res_info["height"], src_left=fullres_srcleft, src_width=fullres_srcwidth, src_top=fullres_srctop, src_height=fullres_srcheight)
    return descale[1:]

def search_for_height(frame, kernel, src_top_start, src_height, height, src_top_step = 0.0125):
    clip = gen_descale_error(frame, width=frame.width, height=height, src_height=src_height, src_top = src_top_start, src_left=0, src_width=frame.width, kernel=kernel)
    for thing in range(1, 20):
        clip += gen_descale_error(frame, width=frame.width, height=height, src_height=src_height, src_top = src_top_start - src_top_step * thing, src_left=0, src_width=frame.width, kernel=kernel)
    return ShowAverage(clip)

def search_for_width(frame, kernel, src_left_start, src_width, width, src_left_step = 0.0125):
    
    clip = gen_descale_error(frame, width=width, height=frame.height, src_height=frame.height, src_top = 0, src_left=src_left_start, src_width=src_width, kernel=kernel)
    for thing in range(1, 20):
        clip += gen_descale_error(frame, width=width, height=frame.height, src_height=frame.height, src_top = 0, src_left=src_left_start - src_left_step * thing, src_width=src_width, kernel=kernel)
    return ShowAverage(clip)

def find_blur(frame, b, c, src_top, src_height, src_left, src_width, width, height, bilinear=False, taps=None, blur_start=0.90, fractional=False):
    clip = gen_descale_error_blur(frame, width=width, height=height, src_height=src_height, src_top = src_top, src_left=src_left, src_width=src_width, blur=blur_start, b=b, c=c, taps=taps, bilinear=bilinear)
    for thing in range(1, 21):
        clip += gen_descale_error_blur(frame, width=width, height=height, src_height=src_height, src_top = src_top, src_left=src_left, src_width=src_width, blur=blur_start + thing * 0.01, b=b, c=c, taps=taps, bilinear=bilinear)
    return ShowAverage(clip)

def look_at_blur(frame, b, c, src_top, src_height, src_left, src_width, width, height, blur_start, bilinear=False):
    if bilinear:
        clip = core.descale.Debilinear(frame, width=width, height=height, src_top = src_top, src_height = src_height, src_width=src_width, src_left=src_left, blur=blur_start)
        for thing in range(1, 21):
            clip += core.descale.Debilinear(frame, width=width, height=height, src_top = src_top, src_height = src_height, src_width=src_width, src_left=src_left, blur=blur_start + 0.01 * thing)
    else:
        clip = core.descale.Debicubic(frame, b=b, c=c, width=width, height=height, src_top = src_top, src_height = src_height, src_width=src_width, src_left=src_left, blur=blur_start)
        for thing in range(1, 21):
            clip += core.descale.Debicubic(frame, b=b, c=c, width=width, height=height, src_top = src_top, src_height = src_height, src_width=src_width, src_left=src_left, blur=blur_start + 0.01 * thing)
    return clip