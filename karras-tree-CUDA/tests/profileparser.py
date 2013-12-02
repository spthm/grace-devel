import numpy as np

class ProfileResult(object):
    """Reads in and saves data for kernel profile results."""
    def __init__(self, open_file):
        super(ProfileResult, self).__init__()

        self.params = {}
        self._read_params(open_file)
        self.levels = np.arange(self.params['start'], self.params['end']+1)
        self._results_start_string = "Will generate"
        self._timing_start_string = "Time for "
        self._timing_end_string = " ms."

        self._move_to_next_data(open_file)
        kernel_names = self._read_kernel_names(open_file)
        self._kernel_names_map = {}
        for (i,s) in enumerate(kernel_names):
            self._kernel_names_map[i] = s
        self.kernel_IDs = range(len(kernel_names))

        self.timings = np.zeros((len(kernel_names),len(self.levels)))
        self._move_to_first_data(open_file)
        for i in range(len(self.levels)):
            self._read_level_results(open_file, i)
            self._move_to_next_data(open_file)

    @classmethod
    def from_filename(cls, filename):
        with open(filename, "r") as f:
            return cls(f)

    def generate_stats(self):
        old = max(self.timings[0])
        for i in self.kernel_IDs:
            new = np.mean(self.timings[i])
            if new < old:
                old = new
                kernel_ID = i
        self.lowest_mean_time = old
        self.fastest_kernel_ID = kernel_ID

        self.kernel_speed_order = range(len(self.timings))
        self.kernel_speed_order.sort(key=lambda i: self.timings[i][0])

    def kernel_name(self, ID):
        return self._kernel_names_map[ID]

    def kernel_ID(self, name):
        for key in self._kernel_names_map:
            if name == self._kernel_names_map[key]:
                return key
        return None

    def _move_to_next_data(self, f):
        pos = f.tell()
        line = f.readline()
        # Find the next data block first, in case we are called from within one.
        while not line.startswith(self._results_start_string) and line != "":
            pos = f.tell()
            line = f.readline()
        f.seek(pos)

        line = f.readline()
        # Now move to the timing data start.
        while not line.startswith(self._timing_start_string) and line != "":
            pos = f.tell()
            line = f.readline()
        f.seek(pos)

    def _move_to_first_data(self, f):
        f.seek(0)
        self._move_to_next_data(f)

    def _read_kernel_names(self, f):
        names = []
        line = f.readline()
        while line.startswith(self._timing_start_string):
            tmp = line.split(":")[0]
            tmp = tmp[len(self._timing_start_string):]
            names.append(tmp)
            line = f.readline()
        return names

    def _read_level_results(self, f, i):
        for ID in self.kernel_IDs:
            line = f.readline()
            tmp = line.split(":")[1]
            tmp = tmp.strip()
            tmp = tmp[:-len(self._timing_end_string)]
            self.timings[ID][i] = tmp

    def _read_params(self, f):
        f.seek(0)
        line = f.readline()
        tmp = line.split(":")[1].strip()
        self.params["device"] = tmp

        line = f.readline()
        tmp = line.split(":")[1].strip()
        self.params["morton block size"] = int(tmp)

        line = f.readline()
        tmp = line.split(":")[1].strip()
        self.params["build block size"] = int(tmp)

        line = f.readline()
        tmp = line.split(":")[1].strip()
        self.params["AABB block size"] = int(tmp)

        line = f.readline()
        tmp = line.split(":")[1].strip()
        self.params["grid size"] = int(tmp)

        line = f.readline()
        tmp = line.split(":")[1].strip()
        self.params["start"] = int(tmp)

        line = f.readline()
        tmp = line.split(":")[1].strip()
        self.params["end"] = int(tmp)

        line = f.readline()
        tmp = line.split(":")[1].strip()
        self.params["iterations"] = int(tmp)

        line = f.readline()
        tmp = line.split(":")[1].strip()
        self.params["seed factor"] = int(tmp)


