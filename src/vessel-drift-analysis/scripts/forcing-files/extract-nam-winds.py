#!python
"""NAM 10 m winds"""
import os
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from pathlib import Path

from nco import Nco

BASEDIR = '/mnt/vault/data/packrat/prod/ncep/nam/alaska_hires/processed/2019/'
OUTDIR = '/mnt/store/data/assets/nps-vessel-spills/forcing_files/nam/'


def extract_uv_surface_noworkie(basedir=BASEDIR, outdir='./', nworkers=10):
    basedir = Path(basedir)
    outdir = Path(outdir)
    files = basedir.glob('**/*.nc')

    nco = Nco()
    with Pool(processes=nworkers) as pool:
        files = list(files)
        for file in files[:2]:
            print(file)
            outfile = outdir / file.name

            # extract winds
            options = [
                '-d height_above_ground4,0',
                '-v wind_u,wind_v'
            ]
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

            # delete dim
            options = [
                '-C',
                '-O',
                '-x',
                '-v height_above_ground4'
            ]
            res = pool.apply_async(
                nco.ncks,
                (),
                {
                    'input': outfile,
                    'output': outfile,
                    'options': options
                }
            )
            print(res.get(timeout=5*60))

            # remove attribute
            options = [
                '-O',
                '-a height_above_groud4'
            ]
            res = pool.apply_async(
                nco.ncwa,
                (),
                {
                    'input': outfile,
                    'output': outfile,
                    'options': options
                }
            )
            print(res.get(timeout=5*60))


def _extract(fname, outdir):
    outdir = Path(outdir)
    outfile = outdir / fname.name

    os.system(f'ncks -4 -O -d height_above_ground4,0 -v wind_u,wind_v {fname} {outfile}')
    os.system(f'ncks -O -C -x -v height_above_ground4 {outfile} {outfile}')
    os.system(f'ncwa -O -a height_above_ground4 {outfile} {outfile}')


def extract_uv_surface(basedir=BASEDIR, outdir=OUTDIR, nworkers=10):
    basedir = Path(basedir)
    outdir = Path(outdir)
    files = basedir.glob('**/*.nc')
    files = list(files)
    files.sort()

    thread_pool = ThreadPool(nworkers)

    for file in files:
        print(file)
        thread_pool.apply_async(_extract, (file, outdir))

    thread_pool.close()
    thread_pool.join()


if __name__ == '__main__':
    extract_uv_surface()
