import tensorflow as tf


class PhysicsModel:
    name = "basemodel"  # Used for the output dictionary

    def __init__(self, indata):
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
    name = "projections"

    def __init__(self, indata, channel, *axes_names):
        info = indata.channel_info[channel]

        self.start = info["start"]
        self.stop = info["stop"]

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
            Project.name,
            "_".join(axes_names) if len(axes_names) else "yield",
        ]

    def make_fun(self, fun_flat, inclusive=True):

        if inclusive:
            out_shape = self.exp_shape[:-1]
            transpose_idxs = self.transpose_idxs[:-1]
        else:
            out_shape = self.exp_shape
            transpose_idxs = self.transpose_idxs

        def proj_fun():
            exp = fun_flat()[self.start : self.stop]
            exp = tf.reshape(exp, out_shape)
            exp = tf.reduce_sum(exp, axis=self.proj_idxs)
            exp = tf.transpose(exp, perm=transpose_idxs)

            return exp

        return proj_fun
