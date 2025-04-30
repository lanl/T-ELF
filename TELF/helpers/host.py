import multiprocessing

def verify_n_jobs(n_jobs):
    """
    Determines what the number of jobs (cores) should be for a given machine. n_jobs can be set
    on [-cpu_count, -1] ∪ [1, cpu_count]. If n_jobs is negative, (cpu_count + 1) - n_jobs cores
    will be used.

    Parameters:
    -----------
    n_jobs: int
        Number of cores requested on this machine

    Returns:
    --------
    n_jobs: int
        Adjusted n_jobs representing real value 

    Raises:
    -------
    ValueError: 
        n_jobs is outside of acceptable range
    TypeError:
        n_jobs is not an int
    """
    cpu_count = multiprocessing.cpu_count()
    if not isinstance(n_jobs, int):
        raise TypeError(f'`n_jobs` must be an int.')

    limit = cpu_count + n_jobs
    if (n_jobs == 0) or (limit < 0) or (2 * cpu_count < limit):
        raise ValueError(f'`n_jobs` must take a value on [-{cpu_count}, -1] or [1, {cpu_count}].')
    
    if n_jobs < 0:
        return cpu_count - abs(n_jobs) + 1
    else:
        return n_jobs