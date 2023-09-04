import os


def check(filepath):
    os.system(f"> {filepath}/__init__.py")
    l = os.listdir(filepath)
    for i in l:
        if i[:2] == "__":
            continue
        if i[-2:] != "py":
            continue
        os.system(f"echo 'from .{i[:-3]} import {i[:-3]}' >> {filepath}/__init__.py")


def check2(filepath):
    os.system(f"> {filepath}/__init__.py")
    l = os.listdir(filepath)
    for i in l:
        if i[:2] == "__":
            continue
        if i[-2:] != "py":
            continue
        os.system(f"echo 'from . import {i[:-3]}' >> {filepath}/__init__.py")


def check_all():
    check("PWR/datasets")
    check("PWR/models")
    check2("scripts")
    check2("PWR/prune")


if __name__ == "__main__":
    check_all()
