import operator

FORMAT_NAMES = {'coo': 'Coordinate Sparse Matrix',
            'csr': 'Compressed Sparse Row Matrix',
            'csc': 'Compressed Sparse Column Matrix'}

# copied from zarr
# See https://github.com/zarr-developers/zarr-python/blob/master/zarr/util.py
def human_readable_size(size):
    if size < 2 ** 10:
        return "%s" % size
    elif size < 2 ** 20:
        return "%.1fK" % (size / float(2 ** 10))
    elif size < 2 ** 30:
        return "%.1fM" % (size / float(2 ** 20))
    elif size < 2 ** 40:
        return "%.1fG" % (size / float(2 ** 30))
    elif size < 2 ** 50:
        return "%.1fT" % (size / float(2 ** 40))
    else:
        return "%.1fP" % (size / float(2 ** 50))


def html_table(arr):
    def bytestr(n):
            if n > 2**10:
                return '%s (%s)' % (n, human_readable_size(n))
            else:
                return str(n)
    table = "<table>"
    table += "<tbody>"
    headings = ["Type", "Format", "Data Type", "Shape", "nnz", "Density", "Read-only",
                "Store", "Compressor", "No. bytes as dense", "No. bytes", "No. bytes stored",
                "Storage ratio", "Chunks initialized"]
    info = [
        '%s.%s' % (type(arr).__module__, type(arr).__name__),
        FORMAT_NAMES[arr.format],
        str(arr.dtype),
        str(arr.shape),
        str(arr.nnz),
        str(arr.nnz / arr.size),
        str(True),
        '%s.%s' % (type(arr._store).__module__, type(arr._store).__name__),
        str(arr.compressor),
        bytestr(arr.shape[0] * arr.shape[1] * arr.dtype.itemsize),
        bytestr(arr.nbytes),
        bytestr(arr.nbytes_stored),
        '%.1f' % (arr.nbytes / arr.nbytes_stored),
        '%s/%s' % (arr.nchunks_initialized, arr.nchunks)
    ]
    for h, i in zip(headings, info):
        table += (
            "<tr>"
            '<th style="text-align: left">%s</th>'
            '<td style="text-align: left">%s</td>'
            "</tr>" % (h, i)
        )
    table += "</tbody>"
    table += "</table>"
    return table
