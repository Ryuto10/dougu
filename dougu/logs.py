import json
import subprocess
import sys
from argparse import Namespace
from datetime import datetime
from os import getcwd, path


def write_log(
        logfile_path: str = None,
        args_file_path: str = None,
        args: Namespace = None,
        comment: str = None
):
    """Write log file

    Args:
        logfile_path (str, optional): Path to output file for log. (Defaults to None)
        args_file_path (str, optional): Path to output file for args. (Defaults to None)
        args (Namespace, optional): argparse.ArgumentParser().parse_args(). (Defaults to None)
        comment (str, optional): custom comments. (Defaults to None)
    """
    write_type = "a" if logfile_path and path.exists(logfile_path) else "w"
    fo = open(logfile_path, write_type) if logfile_path else None
    if write_type == "a":
        print("-" * 50, file=fo)

    print("# Basic Information", file=fo)
    print("## Time", file=fo)
    print("\t", datetime.today().strftime("%Y/%m/%d %H:%M:%S"), file=fo)
    print("## Input Command", file=fo)
    print("\t", " ".join(sys.argv), file=fo)
    print("## The directory of the running script.", file=fo)
    print("\t", getcwd(), file=fo)
    print("## Version", file=fo)
    print("\t", sys.version.replace("\n", "\n\t"), file=fo)
    if comment:
        print("## Comment", file=fo)
        print("\t", comment, file=fo)

    print("\n# pip", file=fo)
    pip_list = subprocess.check_output("pip list", shell=True).decode('utf-8')
    print("\t{}".format(pip_list.replace("\n", "\n\t")), file=fo)

    if path.exists('.git'):
        git_log = subprocess.check_output(
            "git log --pretty=fuller | head -7", shell=True).decode('utf-8')
        git_diff = subprocess.check_output(
            "git diff HEAD " + sys.argv[0], shell=True).decode('utf-8')
        print("\n# Git", file=fo)
        print("## log", file=fo)
        print("> {}  ".format(git_log.replace("\n", "  \n> ")), file=fo)
        print("## diff", file=fo)
        print("> \ {}  ".format(git_diff.replace("\n", "  \n> \ ")), file=fo)

    if args:
        print("\n# Argparse", file=fo)
        for k, v in args.__dict__.items():
            v = path.abspath(v) if type(v) == str and path.exists(v) else v
            print("\t{} : {}  ".format(k, v), file=fo)
        if args_file_path:
            with open(args_file_path, "w") as args_fo:
                json.dump(args.__dict__, args_fo)
    print("", file=fo)
    fo.close()
