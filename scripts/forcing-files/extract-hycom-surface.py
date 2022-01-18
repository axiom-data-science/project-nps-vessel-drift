#!python
"""Extract HYCOM surface u, v files"""
import os
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from pathlib import Path

from nco import Nco

BASEDIR = '/mnt/vault/data/packrat/prod/hycom/orghycom/gofs31/glby/processed/2019/'
OUTDIR = '/mnt/store/data/assets/nps-vessel-spills/forcing-files/hycom/final-files'


def extract_uv_surface_noworkie(basedir=BASEDIR, outdir='./', nworkers=10):
    basedir = Path(basedir)
    outdir = Path(outdir)
    files = basedir.glob('**/*.nc')

    nco = Nco()
    options = [
        '-d depth,0',
        '-d lon,160.0,220.0',
        '-d lat,45.0,75.0',
        '-v water_u,water_v,water_temp,salinity,sic,siu,siv',
        '-L 4'
    ]
    with Pool(processes=nworkers) as pool:
        files = list(files)
        for file in files[:10]:
            print(file)
            outfile = outdir / file.name

            res = pool.apply_async(
                nco.ncks,
                (),
                {
                    'input': file,
                    'output': outfile,
                    'options': options
                }
            )
            print(res.get(timeout=5*60))


def _extract(fname, outdir):
    outdir = Path(outdir)
    outfile = outdir / fname.name

# ugh - failing for some reason.
#    subproc = subprocess.Popen(
#        f'ncks -d lat,45.0,75.0 -d lon,160.0,220.0 -d depth,0 -v water_u,water_v {fname} {outfile}',
#        stdout=subprocess.PIPE,
#        stderr=subprocess.PIPE,
#        shell=True
#    )
#    stdout, stderr = subproc.communicate()
    os.system(f'ncks -d lat,45.0,75.0 -d lon,160.0,220.0 -d depth,0 -v water_u,water_v,water_temp,salinity {fname} {outfile}')


def extract_uv_surface(basedir=BASEDIR, outdir=OUTDIR, nworkers=10):
    basedir = Path(basedir)
    outdir = Path(outdir)
    files = basedir.glob('**/*.nc')
    files = list(files)

    thread_pool = ThreadPool(nworkers)

    for file in files:
        print(file)
#        _extract(file, outdir)

        thread_pool.apply_async(_extract, (file, outdir))

    thread_pool.close()
    thread_pool.join()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'source_dir',
        type=Path,
        default=BASEDIR,
        help='path to directory with hycom files'
    )
    parser.add_argument(
        'output_dir',
        type=Path,
        default=OUTDIR,
        help='path to directory to save output files'
    )
    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=True)
    extract_uv_surface(args.source_dir, args.output_dir)
