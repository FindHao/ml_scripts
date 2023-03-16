#!/usr/bin/env python3
from collections import defaultdict
import re
import subprocess
import requests
from bs4 import BeautifulSoup
from pkginfo import Wheel
import argparse
import os
from urllib.parse import unquote, quote


# The base URL for the PyTorch nightly builds
base_url = None
pkgs = ["torchdata", "torchvision", "torchtext", "torchaudio", "torch", "pytorch-triton"]
# raw dependency data deps = {pkg: [dep1, dep2=XX] }
deps = {}
filter_deps = {}
script_path = os.path.dirname(os.path.realpath(__file__))
down_path = None
python_version=None
platform="linux"

def init(cuda_version, py_version):
    global base_url
    global down_path
    global python_version
    python_version = py_version
    base_url = "https://download.pytorch.org/whl/nightly/%s" % cuda_version
    down_path = "%s/.downloads/%s" % (script_path, cuda_version)
    # check if '.downloads/cu118' folder exists, if not create it
    if not os.path.exists(down_path):
        os.makedirs(down_path)
    print("Downloading packages to %s" % down_path)

def get_download_url(package_name, date_str):
    # Construct the URL for the package
    url = "%s/%s" % (base_url, package_name)
    print("Downloading %s from %s" % (package_name, url))
    response = requests.get(url)
    html = response.text
    # Parse the HTML page
    soup = BeautifulSoup(html, 'html.parser')
    #   find all a tags
    links = soup.find_all('a')
    # Find the link to the wheel file
    link = soup.find('a', href=True)
    # Extract the download URL for the wheel file
    download_url = None
    # find the url with the date
    for link in links:
        if link['href'].find(date_str) != -1 and link['href'].find(python_version) != -1 and link['href'].find(platform) != -1:
            download_url = "%s%s" %("https://download.pytorch.org", link['href'] )
    return download_url


def download_file(url, force=False):
    # obtain the file name from the URL
    file_name = url.split("/")[-1]
    file_name = unquote(file_name)
    file_path = '%s/%s' % (down_path, file_name)
    # check if the file already exists
    if not force:
        try:
            with open(file_path, "r") as f:
                print(
                    "File %s already exists, if you want to overwrite it, use the --force flag" % file_name)
                return file_name
        except FileNotFoundError:
            pass
    result = subprocess.run(["wget", '--continue', '-P', down_path,  url])
    if result.returncode == 0:
        print("File downloaded successfully")
        return file_name
    else:
        print("Error downloading file")
        exit(1)

def download_pytorch_triton(force=False):
    if 'torch' not in filter_deps or 'pytorch-triton' not in filter_deps['torch']:
        return False
    print("Downloading pytorch-triton")
    triton_version = filter_deps["torch"]["pytorch-triton"]
    download_url = get_download_url("pytorch-triton", quote(triton_version))
    if download_url is None:
        print("ERROR -> Could not find a package for %s on %s" % (pkg, triton_version))
        return False
    if download_file(download_url, force=force):
        return True
    else:
        return False


def parse_dependencies(dependencies):
    torch_deps = {}
    for dependency in dependencies:
        # ['typing-extensions', 'sympy', 'networkx', 'pytorch-triton (==2.0.0+0d7e753227)', "pytorch-triton (==2.0.0+0d7e753227) ; extra == 'dynamo'", "jinja2 ; extra == 'dynamo'", "opt-einsum (>=3.3) ; extra == 'opt-einsum'"]
        match = re.match(r"(.+?) \([>=<]*(\d.+)\)", dependency)
        if match:
            package_name = match.group(1)
            package_version = match.group(2)
            # print(f"Package: {package_name}, Version: {package_version}")
            if package_name in pkgs:
                torch_deps[package_name] = package_version
    return torch_deps

def summary(dependency_outputs, pytorch_triton_success):
    if dependency_outputs:
        max_len = max([len(line) for line in dependency_outputs]) + 4
    else:
        max_len = 0
    max_len = max(max_len, 100)
    print("="*(max_len + 4))
    print(('| {:'+str(max_len)+'s} |').format("Summary"))
    print("="*(max_len + 4))
    global filter_deps
    downloaded = set(filter_deps.keys())
    missed = set(pkgs) - downloaded
    if pytorch_triton_success:
        missed -= set(["pytorch-triton"])
        downloaded.add("pytorch-triton")
    print(('| {:'+str(max_len)+'s} |').format("Downloaded Packages:"))
    print(('| {:'+str(max_len)+'s} |').format(", ".join(downloaded)))
    if missed:
        print(('| {:'+str(max_len)+'s} |').format("Missed in full list:"))
        print(('| {:'+str(max_len)+'s} |').format(", ".join(missed)))
    if dependency_outputs:
        print("-"*(max_len + 4))
        print(('| {:'+str(max_len)+'s} |').format("Dependency Issues:"))
        print(('| {:'+str(max_len)+'s} |').format("WARNING: The following packages have dependencies that are not the same version as the package"))
        for line in dependency_outputs:
            print(('| {:'+str(max_len)+'s} |').format(line))
    print("="*(max_len + 4))

def check_dependencies(date_str):
    outputs = []
    for pkg in filter_deps:
        for dep_pkg in filter_deps[pkg]:
            # print(f"Package: {pkg}, Dependency: {dep}")
            dep_pkg_version = filter_deps[pkg][dep_pkg]
            if dep_pkg != "pytorch-triton" and dep_pkg_version.find(date_str) == -1:
                outputs.append(f"Package: {pkg}, Dependency: {dep_pkg} version: {dep_pkg_version}")
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default="20230102",
                        help="date for which you want to download PyTorch. by default it is 20230102")
    parser.add_argument("--cuda", type=str, default="cu118",
                        help="cuda version for which you want to download PyTorch. by default it is cu118")
    parser.add_argument("--python", type=str, default="cp310",
                        help="python version for which you want to download PyTorch. by default it is cp310")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing files")
    # add arguments for pkg name 
    parser.add_argument("--pkgs", type=str, default="torchdata,torchvision,torchtext,torchaudio,torch,pytorch-triton",
                        help="name of the package you want to download")
    args = parser.parse_args()
    init(args.cuda, args.python)
    date_str = args.date
    if args.pkgs:
        pkgs = args.pkgs.split(",")
    for pkg in pkgs:
        if pkg =="pytorch-triton":
            continue
        # Get the download URL for the package
        download_url = get_download_url(pkg, date_str)
        if download_url is None:
            print("ERROR -> Could not find a package for %s on %s" % (pkg, date_str))
            continue
        # Download the package
        file_name = download_file(download_url, force=args.force)
        file_path = '%s/%s' % (down_path, file_name)
        # print dependecies
        deps[pkg] = Wheel(file_path).requires_dist
        print("Dependencies for %s: %s" % (pkg, deps[pkg]))
        torch_deps = parse_dependencies(deps[pkg])
        filter_deps[pkg] = torch_deps
    dependency_output = check_dependencies(date_str)
    pytorch_trition_success = download_pytorch_triton(args.force)
    summary(dependency_output, pytorch_trition_success)
    
