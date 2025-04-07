import os

import cython

from libc.stdlib cimport free, malloc

cdef extern from "sys/types.h":
    ctypedef long off_t
cdef extern from "sys/sendfile.h":
    ssize_t sendfile(int out_fd, int in_fd, off_t * offset, size_t count)

def merge_fds(
    int merged_fd, _component_xyts_files, int merged_nt, int merged_ny, _local_nxs, _local_nys, _y0s
):
    ''' Merge a list of XYTS files by copying their values in order.

    To merge the files we copy the velocity values in each component, for each
    timestep, in the order they would fit in the merged domain. That is if four
    files tile the domain like so:

            ┌───────────┬──────────┐
            │***********│##########│
            │!!!!!!!!!!!│++++++++++│
            │    f1     │    f2    │
            │           │          │
            ├───────────┼──────────┤
            │$$$$$$$$$$$│%%%%%%%%%%│
            │           │          │
            │    f3     │    f4    │
            │           │          │
            │           │          │
            └───────────┴──────────┘

    Then they are concatenated in the output domain as:

    ***********##########!!!!!!!!!!!++++++++++ ... $$$$$$$$$$$%%%%%%%%%%

    It is assumed _component_xyts_files is a list of xyts files sorted by their
    top left corner.
    '''
    cdef int float_size, cur_timestep, cur_component, cur_y, i, y0, local_ny, local_nx, xyts_fd, n_files
    cdef int *component_xyts_files, *local_nxs, *local_nys, *y0s

    n_files = len(_component_xyts_files)

    # The lists _component_xyts_files, ... are CPython lists.
    # If we access these inside our copying loop everything becomes very slow
    # because we need to go to the Python interpreter. So we first copy each
    # list into an equivalent C list.
    component_xyts_files = <int *> malloc(len(_component_xyts_files) * cython.sizeof(int))
    local_nxs = <int *> malloc(len(_local_nxs) * cython.sizeof(int))
    local_nys = <int *> malloc(len(_local_nys) * cython.sizeof(int))
    y0s = <int *> malloc(len(_y0s) * cython.sizeof(int))
    # NOTE: we cannot use memcpy() here because CPython lists are not continuous
    # chunks of memory as they are in C.
    for i in range(len(_component_xyts_files)):
        component_xyts_files[i] = _component_xyts_files[i]
        local_nxs[i] = _local_nxs[i]
        local_nys[i] = _local_nys[i]
        y0s[i] = _y0s[i]
    for cur_timestep in range(merged_nt):
        for cur_component in range(3):  # for each component
            for cur_y in range(merged_ny):
                for i in range(n_files):
                    y0 = y0s[i]
                    local_ny = local_nys[i]
                    local_nx = local_nxs[i]
                    xyts_fd = component_xyts_files[i]
                    if y0 > cur_y:
                        break
                    if cur_y >= y0 + local_ny:
                        continue
                    # By passing NULL as the offset, sendfile() will read from
                    # the current position in xyts_fd
                    sendfile(merged_fd, xyts_fd, NULL, local_nx * 4)
    free(component_xyts_files)
    free(local_nxs)
    free(local_nys)
    free(y0s)
