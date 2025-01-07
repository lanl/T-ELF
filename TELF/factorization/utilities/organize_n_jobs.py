import multiprocessing
import GPUtil
import warnings
import numpy as np

def organize_devices(n_jobs, device, use_gpu):

    if not use_gpu and isinstance(device, int):
        return [device]

    resources = len(GPUtil.getGPUs())

     # single positive integer for device is passed
    if isinstance(device, int) and device >= 0:

        if device > (resources - 1):
            warnings.warn(f"Unknown device {device} requested. Setting it to {resources - 1}")
            device = resources - 1

        device = [device]
    
    # single negative integer for device is passed
    elif isinstance(device, int) and device < 0:
        set_devices = n_jobs + (device + 1)
        device = np.arange(0, set_devices, 1)
    
    # list of devices is pased but there is more devices than number of resources
    elif isinstance(device, list) and len(device) > resources:
        device = np.arange(0, resources, 1)
        warnings.warn(f"More devices than existing GPUs passed. Setting the device to {device}.")
    
    device_temp = []
    device_unknown_flag = False
    for dd in device:
        if dd <= (resources - 1):
            device_temp.append(dd)
        else:
            device_unknown_flag = True

    device = device_temp
    if device_unknown_flag:
        warnings.warn(f"Non existing devices requested. Setting device to {device}.")

    return device

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