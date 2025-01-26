import os

import h5py
import hist
import narf.ioutils


def make_pickle_proxies(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = make_pickle_proxies(v)
        return obj
    elif isinstance(obj, hist.Hist):
        return narf.ioutils.H5PickleProxy(obj)
    else:
        return obj


class Workspace:
    def __init__(self, output_format="narf"):
        self.output_format = output_format

        if output_format == "narf":
            # self.pack = narf.ioutils.H5PickleProxy
            # self.unpack = lambda x: x.get()
            self.dump = lambda k, v, f: narf.ioutils.pickle_dump_h5py(
                k, make_pickle_proxies(v), f
            )
            self.extension = "hdf5"
        elif output_format == "h5py":
            # self.pack = lambda x: x
            # self.unpack = lambda x: x
            self.dump = lambda name, obj, fout: fout.create_dataset(name, obj)
            self.extension = "h5"
            raise NotImplementedError(
                f"Writing to vanilla h5py is under development and not yet supported"
            )
        else:
            raise ValueError(f"Unknown output format {output_format}")

    def write(self, output, results, postfix=None, meta={}):
        outfolder = os.path.dirname(output)
        if outfolder:
            if not os.path.exists(outfolder):
                print(f"Creating output folder {outfolder}")
                os.makedirs(outfolder)

        if "." not in output and output.split(".")[-1]:
            output += f".{self.extension}"

        if postfix is not None:
            parts = output.rsplit(".", 1)
            output = f"{parts[0]}_{postfix}.{parts[1]}"

        print(f"Write output file {output}")
        with h5py.File(output, "w") as fout:
            self.dump("results", results, fout)
            self.dump("meta", meta, fout)
