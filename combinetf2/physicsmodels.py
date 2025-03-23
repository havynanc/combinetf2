import tensorflow as tf


class PhysicsModel:
    """
    A class to output histograms without any transformation.

    Parameters
    ----------
    """

    def __init__(self, indata):
        self.name = "basemodel"  # Used for the output dictionary
        # channel info of the resulting histogram, needed for the writing
        self.channel_info = indata.channel_info
        # if data histograms are stored or not, and if chi2 is calculated
        self.has_data = True
        self.normalize = False
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

        def proj_fun():
            exp = fun_flat()[self.start : self.stop]
            if self.normalize:
                norm = tf.reduce_sum(exp)
                exp /= norm
            exp = tf.reshape(exp, out_shape)
            exp = tf.reduce_sum(exp, axis=self.proj_idxs)
            exp = tf.transpose(exp, perm=transpose_idxs)

            return exp

        return proj_fun

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


models = {
    "basemodel": lambda *args: PhysicsModel(*args),
    "project": lambda *args: Project(*args, normalize=False),
    "normalize": lambda *args: Project(*args, normalize=True),
}
