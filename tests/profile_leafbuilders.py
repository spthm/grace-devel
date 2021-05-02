from glob import glob

import matplotlib.pyplot as plt

# hard-code profiler names, leaf/node/tracing data
# hard-code gadget files and corresponding redshifts
#
# Final output:
#   for each N_particles
#     for each redshift
#       for each profiled measurement:
#         for each branch:
#           for each device:
#             measurement
#
# data['0128/Data_XXX'/'0256/Data_XXX']['leaf/node/intersect/full/sort']['branch']['device']
#
# get branch names from glob
# get gadget file names from branch names + profilers, or just hardcode them

class NestedDict(dict):
    """A dictionary where any d[sequence][of][keys] is always valid.

    Missing keys have a new instance of this class set as their value.
    """

    def __missing__(self, key):
        value = self[key] = self.__class__()
        return value


# Schema entry format:
#   tag: [line-identifying string,
#         line-splitting string,
#         index or slice of resulting split,
#         function to convert string to desired output type]
generic_schemas = {
    'fname': ['Gadget file', '/', slice(-2, None), str],
    'npars': ['Number of particles', ' ', -1, int],
    'max_per_leaf': ['Max particles per leaf', ' ', -1, int],
    'device': ['device', ' ', -1, str]
    }

profile_schemas = {
    'profile_tree_gadget': {'leaftime': ['building leaves', ' ', -2, float],
                            'nodetime': ['building nodes', ' ', -2, float]},
    'profile_trace_gadget': {'intersecttime:': ['hit count', ' ', -2, float],
                             'fulltime': ['full tracing', ' ', -2, float]
                             'sorttime': ['sort-by-distance', ' ', -2, float]}
    }

def extract_tags(fname):
    for tag in profile_schemas:
        if tag in fname:
            profiler_tag = tag
            break
    else:
        raise ValueError("fname " + fname + " has no schema")

    branch_tag = fname.lstrip(profiler_tag).rstrip(profiler_tag)

    return branch_tag, profiler_tag

def extract(line, schema):
    signature, splitter, index, convert = schema
    if signature not in line:
        raise ValueError("Line does not match this schema")

    # 'index' may be an instance of slice()
    string = ''.join(line.split(splitter)[index])
    return convert(string)

def get_chunks(fname):
    lines = []
    chunk_count = 0
    chunk_signature, _, _, _ = generic_schemas['fname']
    with open(fname) as f:
        for line in f:
            if chunk_signature in line:
                chunk_count += 1
            if line != '\n':
                lines.append(line)

    line_count = len(lines)
    chunk_size = line_count / chunk_count
    if line_count % chunk_count != 0:
        raise ValueError("Chunks composed of different numbers of lines")

    chunks = []
    for i in range(chunk_count):
        begin, end = i * chunk_size, (i + 1) * chunk_size
        chunk_lines = lines[begin:end]
        chunks.append(''.join(chunk_lines))

    return chunks

def parse_chunk(chunk, schemas):
    chunk_data = dict.fromkeys(schemas.iterkeys())
    lines = chunk.split('\n')
    for line in lines:
        for tag, schema in schemas.iteritems():
            signature = schema[0]
            if signature in line:
                chunk_data[tag] = extract(line, schema)

def parse(fname):
    file_data = NestedDict()

    branch_tag, profiler_tag = extract_tags(fname)
    profile_schema = profile_schemas[profiler_tag]

    chunks = get_chunks(fname)
    for chunk in chunks:
        chunk_params = parse_chunk(chunk, generic_schema)
        chunk_data = parse_chunk(chunk, profile_schema)
        g_fname = chunk_params['fname']
        device = chunk_params['device']

        for measurement_tag, measurement in chunk_data:
            file_data[g_fname][measurement_tag][branch_tag][device] = measurement

if __name__ == '__main__':
    profile_data = {}

    for (tag, schemas) in profile_schemas.iteritems():
        files = glob(tag + "/*.log")
        print "Tag:", tag
        for fname in files:
            print fname
            # file_data = parse(fname)
            # profile_data.update(file_data)

    plots = NestedDict()
    for (g_fname, file_data) in profile_data.iteritems():
        redshift, N = gadget_params(g_fname)
        for (measurement_tag, measurement_data) in file_data.iteritems():
            plt.figure()
            plt.title(measuremnt_tag + " at z = " + redshift + ", N = ")

            # One group of bars per branch.
            group_loc = 0
            # Each group takes up 80% of the available width, and each
            # per-device bar takes up an even share of that 80%.
            bar_width = 0.8 / len(measurement_data)
            for (branch_tag, branch_data): in measurement_data.iteritems():
                # One bar per device in each group.
                dev_loc = 0
                for (device_tag, measurement) in branch_data.iteritems():
                    plt.bar(dev_loc, measurement, bar_width)
                    dev_loc += width
                 group_loc += 1

    plt.show()

#
#
# T
# i
# m
# e
#
#    || dev A | dev B ||| dev A | dev B ||| dev A | dev B ||| dev A | dev B |||
#    |     branch A    |     branch B    |     branch C    |    branch D     |
#                      LEAF BUILD TIME for N = 128^3, z = 8
