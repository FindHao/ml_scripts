import re
import subprocess
import requests
from bs4 import BeautifulSoup
from pkginfo import Wheel
import argparse
import os
from urllib.parse import unquote, quote


# The base URL for the PyTorch nightly builds
base_url = "https://download.pytorch.org/whl/nightly/cu116"
pkgs = ["torchdata", "torchvision", "torchtext", "torchaudio", "torch", "pytorch-triton"]
# raw dependency data deps = {pkg: [dep1, dep2=XX] }
deps = {}
filter_deps = {}
script_path = os.path.dirname(os.path.realpath(__file__))
# check if '.downloads/cu116' folder exists, if not create it
if not os.path.exists('%s/.downloads/cu116' % script_path):
    os.makedirs('%s/.downloads/cu116' % script_path)
down_path = "%s/.downloads/cu116" % script_path
print("Downloading packages to %s" % down_path)
python_version="cp38"
platform="linux"


def get_download_url(package_name, date_str):
    # Construct the URL for the package
    url = "%s/%s" % (base_url, package_name)
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
    print("Downloading pytorch-triton")
    triton_version = filter_deps["torch"]["pytorch-triton"]
    download_url = get_download_url("pytorch-triton", quote(triton_version))
    if download_url is None:
        print("Could not find a package for %s on %s" % (pkg, triton_version))
    download_file(download_url, force=force)


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

def check_dependencies(date_str):
    for pkg in filter_deps:
        for dep_pkg in filter_deps[pkg]:
            # print(f"Package: {pkg}, Dependency: {dep}")
            dep_pkg_version = filter_deps[pkg][dep_pkg]
            if dep_pkg != "pytorch-triton" and dep_pkg_version.find(date_str) == -1:
                print(f"Package: {pkg}, Dependency: {dep_pkg} is not the same version as the package")
                print(f"Package: {pkg}, Dependency: {dep_pkg} version: {dep_pkg_version}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default="20230102",
                        help="date for which you want to download PyTorch")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing files")
    # add arguments for pkg name 
    parser.add_argument("--pkgs", type=str, default="torchdata,torchvision,torchtext,torchaudio,torch,pytorch-triton",
                        help="name of the package you want to download")
    args = parser.parse_args()
    date_str = args.date
    if args.pkgs:
        pkgs = args.pkgs.split(",")
    for pkg in pkgs:
        if pkg =="pytorch-triton":
            continue
        # Get the download URL for the package
        download_url = get_download_url(pkg, date_str)
        if download_url is None:
            print("Could not find a package for %s on %s" % (pkg, date_str))
            continue
        # Download the package
        file_name = download_file(download_url, force=args.force)
        file_path = '%s/%s' % (down_path, file_name)
        # print dependecies
        deps[pkg] = Wheel(file_path).requires_dist
        print("Dependencies for %s: %s" % (pkg, deps[pkg]))
        torch_deps = parse_dependencies(deps[pkg])
        filter_deps[pkg] = torch_deps
    print("=====================================")
    check_dependencies(date_str)
    download_pytorch_triton(args.force)
    
