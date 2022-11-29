from utils.save_load import find_by_job_number
import argparse

parser = argparse.ArgumentParser()
jobs = [233168, 233168]
parser.add_argument('--jobs', '-j', default=jobs, type=int)

args = parser.parse_args()

j_ = args.jobs

m_ = [find_by_job_number(j, load_net=True, load_state=True) for j in j_]

means = [m['net'].encoder.prior.mean for m in m_]

print((means[0] - means[1]).pow(2).sum())
