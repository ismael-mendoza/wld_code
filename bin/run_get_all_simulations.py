#!/usr/bin/env python3
import argparse
import os

from src import utils

survey_name = 'LSST'


def main(args):
    if args.small:
        gs = ['-.02', '-.015', '-.01', '-.005', '.005', '.01', '.015', '0.02']
    else:
        gs = ['-0.1', '-0.05', '-.02', '-.015', '-.01', '-.005', '.005', '.01', '.015', '0.02', '0.05', '0.1']

    for g in gs:
        project_final_name = f"{args.project}{survey_name}-g1_{int(float(g) * 1000)}-g2_0"
        output_file = os.path.join(utils.data_dir, project_final_name, "output-final.txt")

        if args.simulate:
            cmd = f'python bin/all-process.py --simulate-all --num-sections 10 --cosmic-shear-g1 {g} ' \
                  f'--cosmic-shear-g2 0 --project {project_final_name} --survey-name {survey_name} ' \
                  f'--max-memory 4096MB --bjob-time 04:00'

        elif args.process:
            cmd = f'python bin/all-process.py --add-noise-all --extract-all --combine ' \
                  f'--num-sections 10 --project {project_final_name} --noise-seed 0 --survey-name {survey_name}'

        else:
            break

        slac_cmd = f'bsub -W 00:50 -M 2000MB -o {output_file} -r "{cmd}"'
        print(slac_cmd)
        os.system(slac_cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulate different regions from a square degree and analyze their '
                                                 'combination in SExtractor.')
    parser.add_argument('--small', action='store_true')
    parser.add_argument('--simulate', action='store_true')
    parser.add_argument('--process', action='store_true')
    parser.add_argument('--project', type=str, default='project')
    pargs = parser.parse_args()
    main(pargs)
