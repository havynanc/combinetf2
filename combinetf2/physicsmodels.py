import re

import tensorflow as tf


class PhysicsModel:
    """
    A class to output histograms without any transformation.
    """

    def __init__(self, indata):
        self.name = "basemodel"  # Used for the output dictionary
        # channel info of the resulting histogram, needed for the writing
        self.channel_info = indata.channel_info
        # if data histograms are stored or not, and if chi2 is calculated
        self.has_data = True
        # a list of strings, the result will be written under these keys
        self.identifiers = []

    def make_fun(self, fun_flat, inclusive=True):
        return fun_flat

    def get_data(self, data):
        fun = self.make_fun(lambda: data)
        return fun(), None


class Project(PhysicsModel):
    """
    A class to project a histogram to lower dimensions.
    Optionally the histogram can be normalized.
    The normalization is done to the integral of all processes or data.

    Parameters
    ----------
    channel_name : str
        Name of the channel. Required.
    axes_names : list of str, optional
        Names of the axes to keep. If empty, the histogram will be projected to a single bin.
    """

    def __init__(self, indata, channel, *axes_names, normalize=False):
        info = indata.channel_info[channel]

        self.start = info["start"]
        self.stop = info["stop"]
        self.normalize = normalize
        self.name = "normalized" if normalize else "projections"

        channel_axes = info["axes"]

        hist_axes = [axis for axis in channel_axes if axis.name in axes_names]

        if len(hist_axes) != len(axes_names):
            raise ValueError(
                f"Hist axes {[h.name for h in hist_axes]} != {axes_names} not found"
            )

        exp_axes = channel_axes.copy()

        # for per process histograms
        exp_axes.append(indata.axis_procs)
        extra_axes = [indata.axis_procs]

        self.exp_shape = tuple([len(a) for a in exp_axes])

        channel_axes_names = [axis.name for axis in channel_axes]
        extra_axes_names = [axis.name for axis in extra_axes]

        axis_idxs = [channel_axes_names.index(axis) for axis in axes_names]

        self.proj_idxs = [i for i in range(len(channel_axes)) if i not in axis_idxs]

        post_proj_axes_names = [
            axis for axis in channel_axes_names if axis in axes_names
        ] + extra_axes_names

        self.transpose_idxs = [
            post_proj_axes_names.index(axis) for axis in axes_names
        ] + [post_proj_axes_names.index(axis) for axis in extra_axes_names]

        self.has_data = not info["masked"]

        self.channel_info = {
            channel: {
                "axes": hist_axes,
                "start": None,
                "stop": None,
            }
        }

        self.identifiers = [
            self.name,
            "_".join(axes_names) if len(axes_names) else "yield",
        ]

    def make_fun(self, fun_flat, inclusive=True):

        if inclusive:
            out_shape = self.exp_shape[:-1]
            transpose_idxs = self.transpose_idxs[:-1]
        else:
            out_shape = self.exp_shape
            transpose_idxs = self.transpose_idxs

        def fun():
            exp = fun_flat()[self.start : self.stop]
            if self.normalize:
                norm = tf.reduce_sum(exp)
                exp /= norm
            exp = tf.reshape(exp, out_shape)
            exp = tf.reduce_sum(exp, axis=self.proj_idxs)
            exp = tf.transpose(exp, perm=transpose_idxs)

            return exp

        return fun

    def get_data(self, data, cov=None):
        val = self.make_fun(lambda: data)()

        shape = self.exp_shape[:-1]
        perm_idxs = self.transpose_idxs[:-1]

        if cov is not None:
            cov = cov[self.start : self.stop, self.start : self.stop]

            cov = tf.reshape(cov, [*shape, *shape])
            for idx in sorted(self.proj_idxs, reverse=True):
                cov = tf.reduce_sum(cov, axis=idx)
                cov = tf.reduce_sum(cov, axis=idx + len(shape) - 1)
            cov = tf.transpose(
                cov, perm=perm_idxs + [i + len(perm_idxs) for i in perm_idxs]
            )

            if self.normalize:
                norm = tf.reduce_sum(val)
                jac = tf.eye(tf.shape(val)[0]) / norm - tf.expand_dims(val, 1) / norm**2
                cov = tf.matmul(jac, tf.matmul(cov, tf.transpose(jac)))

            var = tf.linalg.diag_part(cov)
        else:
            var = data[self.start : self.stop]

            if self.normalize:
                norm = tf.reduce_sum(var)
                var = var / (norm**2)
                # Additional variance from the normalization uncertainty
                var = var + var**2

            var = tf.reshape(var, shape)
            var = tf.reduce_sum(var, axis=self.proj_idxs)
            var = tf.transpose(var, perm=perm_idxs)

        return val, var


class Term:
    def __init__(
        self,
        indata,
        channel,
        processes=[],
        selections={},
    ):
        info = indata.channel_info[channel]

        self.start = info["start"]
        self.stop = info["stop"]

        channel_axes = info["axes"]

        channel_axes_names = [a.name for a in channel_axes]

        self.has_data = not info["masked"] and len(processes) == 0

        self.exp_shape = tuple([len(a) for a in channel_axes])

        if processes is not None:
            if any(p not in indata.procs.astype(str) for p in processes):
                raise RuntimeError(
                    f"Not all selection processes found in channel. Selection processes: {processes}, Channel axes: {indata.procs}"
                )
            self.proc_idxs = [
                i for i, p in enumerate(indata.procs.astype(str)) if p in processes
            ]
        else:
            self.proc_idxs = None

        if selections is not None:
            if any(k not in channel_axes_names for k in selections.keys()):
                raise RuntimeError(
                    f"Not all selection axis found in channel. Selection axes: {selections.keys()}, Channel axes: {channel_axes_names}"
                )
            self.selections = tuple(
                [
                    selections[n] if n in selections.keys() else slice(None)
                    for i, n in enumerate(channel_axes_names)
                ]
            )
            channel_axes = [c for c in channel_axes if c.name not in selections.keys()]
            self.selection_idxs = [
                i for i, n in enumerate(channel_axes_names) if n in selections.keys()
            ]
        else:
            self.selections = None
            self.selection_idxs = None

        self.channel_axes = channel_axes


class Ratio(PhysicsModel):
    """
    A class to compute ratios of channels, processes, or bins.
    Optionally the numerator and denominator can be normalized.

    Parameters
    ----------
        indata: Input data used for analysis (e.g., histograms or data structures).
        num_channel: str
            Name of the numerator channel.
        den_channel: str
            Name of the denominator channel.
        num_processes: list of str, optional
            List of process names for the numerator channel. Defaults to None, meaning all processes will be considered.
            Selected processes are summed before the ratio is computed.
        den_processes: list of str, optional
            Same as num_processes but for denumerator
        num_selection: dict, optional
            Dictionary specifying selection criteria for the numerator. Keys are axis names, and values are slices or conditions.
            Defaults to an empty dictionary meaning no selection.
            E.g. {"charge":0, "ai":slice(0,2)}
            Selected axes are summed before the ratio is computed. To integrate over one axis before the ratio, use `slice(None)`
        den_selection: dict, optional
            Same as num_selection but for denumerator
        normalize: bool, optional
            Whether to normalize the numerator and denominator before the ratio. Defaults to False.
    """

    def __init__(
        self,
        indata,
        num_channel,
        den_channel,
        num_processes=[],
        den_processes=[],
        num_selection={},
        den_selection={},
        normalize=False,
    ):
        self.normalize = normalize
        self.name = "normratios" if normalize else "ratios"

        self.num = Term(indata, num_channel, num_processes, num_selection)
        self.den = Term(indata, den_channel, den_processes, den_selection)

        self.has_data = self.num.has_data and self.den.has_data

        self.need_processes = len(num_processes) or len(
            den_processes
        )  # the fun_flat will be by processes

        # The output of ratios will always be without process axis
        self.skip_per_process = True

        if self.num.channel_axes != self.den.channel_axes:
            raise RuntimeError(
                "Channel axes for numerator and denominator must be the same"
            )

        hist_axes = self.num.channel_axes

        if num_channel == den_channel:
            channel = num_channel
        else:
            channel = f"{num_channel}_{den_channel}"

        self.has_processes = False  # The result has no process axis

        self.channel_info = {
            channel: {
                "axes": hist_axes,
                "start": None,
                "stop": None,
            }
        }

        self.identifiers = [
            self.name,
            "_".join([a.name for a in hist_axes]) if len(hist_axes) else "yield",
        ]

    def make_fun(self, fun_flat, inclusive=True):

        def fun():
            exp_full = fun_flat()
            exp_num = exp_full[self.num.start : self.num.stop]
            exp_den = exp_full[self.den.start : self.den.stop]

            if not inclusive:
                if self.num.proc_idxs:
                    exp_num = tf.gather(exp_num, indices=self.num.proc_idxs, axis=-1)
                if self.den.proc_idxs:
                    exp_den = tf.gather(exp_den, indices=self.den.proc_idxs, axis=-1)
                exp_num = tf.reduce_sum(exp_num, axis=-1)
                exp_den = tf.reduce_sum(exp_den, axis=-1)

            exp_num = tf.reshape(exp_num, self.num.exp_shape)
            exp_den = tf.reshape(exp_den, self.den.exp_shape)

            if self.num.selections:
                exp_num = exp_num[self.num.selections]
                exp_num = tf.reduce_sum(exp_num, axis=self.num.selection_idxs)
            if self.den.selections:
                exp_den = exp_den[self.den.selections]
                exp_den = tf.reduce_sum(exp_den, axis=self.den.selection_idxs)

            if self.normalize:
                norm_num = tf.reduce_sum(exp_num)
                exp_num /= norm_num
                norm_den = tf.reduce_sum(exp_den)
                exp_den /= norm_den

            # exp_den = tf.where(exp_den <= 0, tf.ones_like(exp_den), exp_den)

            exp = exp_num / exp_den

            return exp

        return fun


def parse_axis_selection(selection_str):
    if selection_str == "None:None":
        return None
    else:
        sel = {}
        for s in re.split(r",(?![^()]*\))", selection_str):
            k, v = s.split(":")
            if "slice" in v:
                sl = slice(
                    *[int(x) if x != "None" else None for x in v[6:-1].split(",")]
                )
            else:
                sl = slice(int(v), int(v) + 1) if v != "None" else slice(None)
            sel[k] = sl
        return sel


def parse_ratio(indata, *args, normalize=False):
    """
    parsing the input arguments into the ratio constructor, is has to be called as
    -m ratio
        <ch num> <ch den>
        <proc_num_0>,<proc_num_1>,... <proc_num_0>,<proc_num_1>,...
        <axis_num_0>:<slice_num_0>,<axis_num_1>,<slice_num_1>... <axis_den_0>,<slice_den_0>,<axis_den_1>,<slice_den_1>...

    Processes selections are optional. But in case on is given for the numerator, the denominator must be specified as well and vice versa.
    Use 'None' if you don't want to select any for either numerator xor denominator.

    Axes selections are optional. But in case one is given for the numerator, the denominator must be specified as well and vice versa.
    Use 'None:None' if you don't want to do any for either numerator xor denominator.
    """

    if len(args) > 2 and ":" not in args[2]:
        procs_num = [p for p in args[2].split(",") if p != "None"]
        procs_den = [p for p in args[3].split(",") if p != "None"]
    else:
        procs_num = []
        procs_den = []

    # find axis selections
    if any(a for a in args if ":" in a):
        sel_args = [a for a in args if ":" in a]
        axis_selection_num = parse_axis_selection(sel_args[0])
        axis_selection_den = parse_axis_selection(sel_args[1])
    else:
        axis_selection_num = None
        axis_selection_den = None

    return Ratio(
        indata,
        args[0],
        args[1],
        procs_num,
        procs_den,
        axis_selection_num,
        axis_selection_den,
        normalize=normalize,
    )


models = {
    "basemodel": lambda *args: PhysicsModel(*args),
    "project": lambda *args: Project(*args, normalize=False),
    "normalize": lambda *args: Project(*args, normalize=True),
    "ratio": lambda *args: parse_ratio(*args, normalize=False),
    "normratio": lambda *args: parse_ratio(*args, normalize=True),
}
