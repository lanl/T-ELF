import multiprocessing
import GPUtil
import warnings

def organize_n_jobs(use_gpu, n_jobs):
    # check if GPUs available if requested
    if use_gpu:

        if len(GPUtil.getGPUs()) <= 0:
            warnings.warn("No GPU found! Using CPUs")
            use_gpu = False

    # if resources requested
    if n_jobs < 0:

        # gpu
        if use_gpu:
            # get the number of GPUs
            resources = len(GPUtil.getGPUs())

        # cpu
        else:
            resources = multiprocessing.cpu_count()
        n_jobs = resources + (n_jobs + 1)

    # 0 or less resources requested
    if n_jobs <= 0:
        raise Exception("Number of GPUs or CPUs must be 1 or more.")

    # too many GPUs requested
    if use_gpu:
        if n_jobs > len(GPUtil.getGPUs()) and use_gpu:
            n_jobs = len(GPUtil.getGPUs())
            warnings.warn(
                "Too mang GPUs requested. Reverting to max available:" + str(n_jobs)
            )
    else:
        # too many CPUs requested
        if n_jobs > multiprocessing.cpu_count() and not use_gpu:
            n_jobs = multiprocessing.cpu_count()
            warnings.warn(
                "Too mang CPUs requested. Reverting to max available:" + str(n_jobs)
            )

    return n_jobs, use_gpu