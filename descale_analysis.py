import vapoursynth as vs
from vapoursynth import core
import functools
from vskernels import Kernel

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

def gen_descale_error(clip: vs.VideoNode, width: float, height: float, kernel: Kernel, src_height: float, src_top: float, src_width: float, src_left: float, thr: float = 0.01) -> vs.VideoNode:
    clip = clip.resize.Point(format=vs.GRAYS, matrix_s='709' if clip.format.color_family == vs.RGB else None)
    descale = kernel.descale(clip, width=width, height=height, src_height=src_height, src_top=src_top, src_width=src_width, src_left=src_left)
    rescale = kernel.scale(descale, width=clip.width, height=clip.height, src_height=src_height, src_top=src_top, src_width=src_width, src_left=src_left)
    diff = core.std.Expr([clip, rescale], f'x y - abs dup {thr} > swap 0 ?').std.Crop(10, 10, 10, 10)
    diff = core.std.Expr([diff], f'x 32 *')
    return diff

def add_mask_value(clip):
    comp_mask = clip.resize.Point(format=vs.GRAYS, matrix_s='709' if clip.format.color_family == vs.RGB else None)
    comp_mask = core.std.Sobel(comp_mask)
    comp_mask = core.std.PlaneStats(comp_mask, prop='Mask')
    clip_wtf = core.std.CopyFrameProps(clip, comp_mask)
    return clip_wtf

def process_descale_settings_dict(clip, descale_settings):
    if "width" not in descale_settings or "height" not in descale_settings:
        Exception("You need to set both height and width!")
    if not isinstance(descale_settings["kernel"], Kernel):
        Exception("'kernel' needs to be a vskernels kernel, e.g. vskernels.Bilinear() etc.")
    kernel = descale_settings["kernel"]
    width = descale_settings["width"]
    height = descale_settings["height"]
    if not isinstance(height, int):
        Exception("height must be an int")
    if not isinstance(width, int):
        Exception("width must be an int")
    if "fractional" in descale_settings:
        descale_type = "fractional"
        if "src_height" in descale_settings or "src_width" in descale_settings or "src_left" in descale_settings or "src_top" in descale_settings:
            Exception("You can't set both fractional and src_ values")
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
    return dict(kernel=kernel, width=width, height=height, src_height=src_height, src_width=src_width, src_top=src_top, src_left=src_left, descale_type=descale_type)

def test_descale_error(clip, descale_settings):
    clip = clip.resize.Point(format=vs.GRAYS)
    clip = add_mask_value(clip)
    from vskernels import Kernel
    #descale_settings needs to be a dict with width, height, "kernel" (a vs-kernels object),
    #and optionally EITHER src_ values or "fractional", a float.
    #src_ values are the descale versions. "fractional" is equivalent to getfnative output.
    descale_settings = process_descale_settings_dict(clip, descale_settings)
    width, height, kernel, src_top, src_height, src_width, src_left = descale_settings["width"], descale_settings["height"], descale_settings["kernel"], descale_settings["src_top"], descale_settings["src_height"], descale_settings["src_width"], descale_settings["src_left"]
    def get_calc(n, f, clip, core):
        diff_raw = f.props['PSAverage']
        mask_value = f.props['MaskAverage']
        if mask_value == 0:
            diff_primary = 0
        else:
            diff_primary = diff_raw / mask_value
        return core.text.Text(clip, str(diff_primary))
    diff = gen_descale_error(clip, width=width, height=height, kernel=kernel, src_top = src_top, src_height = src_height, src_width=src_width, src_left=src_left)
    diff = core.std.PlaneStats(diff, prop='PS')
    return core.std.FrameEval(diff, functools.partial(get_calc, clip=diff, core=vs.core), prop_src=[diff])

def get_descale_ranges(clip, txtfilename, kernels, ind_error_thr = 0.01, avg_error_thr = 0.006, generate_txt_files=True):
    if not isinstance(txtfilename, str):
        raise TypeError("txtfilename must be a string")
    clipdown = core.resize.Bicubic(clip, 854, 480, format=vs.YUV420P8)
    clipdown = core.wwxd.WWXD(clipdown)
    clip = core.resize.Point(clip, format=vs.GRAYS)
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
        descale_kernel = kernel["kernel"]
        kernel_args = descale_kernel.get_scale_args("asdf")
        if 'filter_param_a' in kernel_args and 'filter_param_b' in kernel_args:
            kernelappend = f"bicubic_{kernel_args['filter_param_a']}_{kernel_args['filter_param_b']}"
        elif 'filter_param_a' in kernel_args:
            kernelappend = f"lanczos{kernel_args['filter_param_a']}"
        else:
            kernelappend = f"bilinear"
        if descale_settings["descale_type"] == "integer":
            kernelappend += f"_{descale_settings['width']}_{descale_settings['height']}"
        elif descale_settings["descale_type"] == "fractional":
            kernelappend += f"_{kernel['fractional']}"
        else:
            kernelappend += f"_{descale_settings['width']}_{descale_settings['height']}_{descale_settings['src_top']}_{descale_settings['src_height']}_{descale_settings['src_left']}_{descale_settings['src_width']}"
        kernel_appends.append(kernelappend)
        diff = gen_descale_error(clip, kernel=descale_settings['kernel'], width=descale_settings['width'], height=descale_settings['height'], src_top=descale_settings['src_top'], src_height=descale_settings['src_height'], src_width=descale_settings['src_width'], src_left=descale_settings['src_left'])
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
                avg_error.append(total_error[m]/frames/bias)
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
    if generate_txt_files == True:
        for m in range(len(total_error)):
            with open("string_" + txtfilename + f"_{kernel_appends[m]}.txt", "w") as x:
                x.write(str(frame_ranges[m]))
        with open(f"string_{txtfilename}_nokernel.txt", "w") as x:
            x.write(str(nokernel_ranges))
    return frame_ranges
