# https://github.com/WilsonWangTHU/NerveNet/blob/master/tool/init_path.py
import os.path as osp
import sys
import datetime


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


_this_dir = osp.dirname(__file__)
running_start_time = datetime.datetime.now()
time = str(running_start_time.strftime("%Y_%m_%d-%X"))
# _base_dir = osp.join(_this_dir, '..')
_base_dir = "/".join(_this_dir.split("/")[:-2]) # since this repo contains the environments made by NerveNet repo
# add_path(_base_dir)

def bypass_frost_warning():
    return 0


def get_base_dir():
    return _base_dir


def get_time():
    return time


def get_abs_base_dir():
    return osp.abspath(_base_dir)
