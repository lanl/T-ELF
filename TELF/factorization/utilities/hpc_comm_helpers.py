import numpy as np
import sys

def signal_workers_exit(comm, n_nodes):
    for job_rank in range(1, n_nodes, 1):
        req = comm.isend(np.array([True]),
                            dest=job_rank, tag=int(f'400{job_rank}'))
        req.wait()

def worker_check_exit_status(rank, comm):
    if comm.iprobe(source=0, tag=int(f'400{rank}')):
        sys.exit(0)

def get_next_job_at_worker(rank, comm, comm_buff_size):
    job_flag = True
    if comm.iprobe(source=0, tag=int(f'200{rank}')):
        req = comm.irecv(buf=bytearray(b" " * comm_buff_size),
                            source=0, tag=int(f'200{rank}'))
        data = req.wait()

    else:
        job_flag = False
        data = {}

    return data, job_flag

def collect_results_from_workers(rank, comm, n_nodes, node_status, comm_buff_size):

    all_results = []
    # collect results at root
    if n_nodes > 1 and rank == 0:
        for job_rank, status_info in node_status.items():
            if node_status[job_rank]["free"] == False and comm.iprobe(source=job_rank, tag=int(f'300{job_rank}')):
                req = comm.irecv(buf=bytearray(b" " * comm_buff_size),
                                    source=job_rank, tag=int(f'300{job_rank}'))
                node_results = req.wait()
                node_status[job_rank]["free"] = True
                node_status[job_rank]["job"] = None
                all_results.append(node_results)

    return all_results

def send_job_to_worker_nodes(comm, target_jobs, node_status):
    scheduled = 0
    available_jobs = list(target_jobs.keys())
    # remove the jobs that are in the nodes from the available jobs
    for _, status_info in node_status.items():
        if status_info["job"] is not None and status_info["free"] is False:
            try:
                _ = available_jobs.pop(available_jobs.index(status_info["job"]))
            except Exception as e:
                print(e)
    
    for job_rank, status_info in node_status.items():
        if len(available_jobs) > 0 and status_info["free"]:
            next_job = available_jobs.pop(0)
            req = comm.isend(target_jobs[next_job],
                                dest=job_rank, tag=int(f'200{job_rank}'))
            req.wait()
            node_status[job_rank]["free"] = False
            node_status[job_rank]["job"] = next_job
            scheduled += 1