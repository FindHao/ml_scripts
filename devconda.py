#!/usr/bin/env python3

"""
devconda.py - Conda Environment Annotation Tool

This script manages comments/annotations for conda environments, helping users
track the purpose and characteristics of different environments. Comments are stored
in a JSON file and organized by hostname, allowing for cross-server sharing.

File format:
    ~/.conda_comment.json - Stored as {hostname: {condaenvname: comment}}

Usage:
    No arguments     - Display comments for the current conda environment
    --list, -l       - List all conda environment comments on the current host
    --set, -s COMMENT - Set a comment for the current environment
    --append, -a COMMENT - Append to the existing comment for the current environment
    --env, -e ENV_NAME - Specify which conda environment to operate on (with --set or --append)

Examples:
    python devconda.py                              # Show current environment comment
    python devconda.py --list                       # List all environment comments
    python devconda.py --set "For data science projects"  # Set current environment comment
    python devconda.py --append "with TensorFlow"   # Append to current environment comment
    python devconda.py --set "ML env" --env ml_env  # Set comment for a specific environment
"""

import argparse
import json
import os
import socket
import subprocess
from pathlib import Path

COMMENT_FILE = os.path.expanduser("~/.conda_comment.json")


def get_current_conda_env():
    """Get the name of the current conda environment."""
    return os.environ.get("CONDA_DEFAULT_ENV", "base")


def get_hostname():
    """Get the current hostname."""
    return socket.gethostname()


def get_conda_envs():
    """Get list of conda environments using conda command line."""
    try:
        # Try using the conda command to list environments
        result = subprocess.run(['conda', 'env', 'list', '--json'],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True,
                                check=True)

        env_json = json.loads(result.stdout)
        # Extract environment names from paths
        envs = []
        for env_path in env_json.get('envs', []):
            env_name = os.path.basename(env_path)
            # Handle base environment which might be named differently
            if env_path == env_json.get('root_prefix'):
                env_name = 'base'
            envs.append(env_name)

        return envs
    except (subprocess.SubprocessError, json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Warning: Failed to get conda environments using command: {e}")
        return []


def load_comments():
    """Load comments from the comments file."""
    if not os.path.exists(COMMENT_FILE):
        return {}

    try:
        with open(COMMENT_FILE, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not parse {COMMENT_FILE}. Using empty comments.")
        return {}
    except Exception as e:
        print(f"Error loading comments: {e}")
        return {}


def save_comments(comments):
    """Save comments to the comments file."""
    try:
        # Ensure parent directory exists
        Path(COMMENT_FILE).parent.mkdir(parents=True, exist_ok=True)

        with open(COMMENT_FILE, 'w') as f:
            json.dump(comments, f, indent=2)
    except Exception as e:
        print(f"Error saving comments: {e}")


def get_comment(env_name=None, hostname=None):
    """Get the comment for a specific environment and hostname."""
    comments = load_comments()
    hostname = hostname or get_hostname()
    env_name = env_name or get_current_conda_env()

    if hostname not in comments:
        return None

    return comments[hostname].get(env_name)


def set_comment(comment, env_name=None, hostname=None, append=False):
    """Set or append a comment for a specific environment and hostname."""
    comments = load_comments()
    hostname = hostname or get_hostname()
    env_name = env_name or get_current_conda_env()

    # Verify the environment exists
    conda_envs = get_conda_envs()
    if env_name not in conda_envs:
        print(
            f"Warning: Environment [{env_name}] not found in conda environments.")
        response = input(
            f"Do you want to add a comment for [{env_name}] anyway? (y/n): ")
        if response.lower() != 'y':
            return

    # Initialize hostname entry if it doesn't exist
    if hostname not in comments:
        comments[hostname] = {}

    # Get the current comment, if any
    current_comment = comments[hostname].get(env_name, "")

    # Set or append the comment
    if append and current_comment:
        comments[hostname][env_name] = current_comment + ", " + comment
    else:
        comments[hostname][env_name] = comment

    save_comments(comments)
    print(
        f"Comment for environment [{env_name}] on host [{hostname}] updated.")


def list_comments(hostname=None):
    """List all comments for a specific hostname."""
    comments = load_comments()
    hostname = hostname or get_hostname()
    conda_envs = get_conda_envs()

    if hostname not in comments or not comments[hostname]:
        print(f"No comments found for host [{hostname}].")
        return

    print(f"[{hostname}]")

    # Get all environments with comments
    commented_envs = comments[hostname].keys()

    # Check for environments with comments
    found_comments = False

    # Print environments with comments
    for env_name in conda_envs:
        if env_name in commented_envs:
            print(f"\t{env_name}: {comments[hostname][env_name]}")
            found_comments = True

    # Print environments with comments that may not exist anymore
    for env_name in commented_envs:
        if env_name not in conda_envs:
            print(
                f"\t{env_name}: {comments[hostname][env_name]} (environment not found)")
            found_comments = True

    if not found_comments:
        print("\tNo comments found.")


def main():
    parser = argparse.ArgumentParser(
        description="Manage conda environment comments.")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List all conda environment comments.")
    parser.add_argument("--set", "-s", metavar="COMMENT",
                        help="Set comment for the current conda environment.")
    parser.add_argument("--append", "-a", metavar="COMMENT",
                        help="Append to the comment for the current conda environment.")
    parser.add_argument("--env", "-e", metavar="ENV_NAME",
                        help="Specify conda environment name.")

    args = parser.parse_args()

    # If --list is specified, list all comments and exit
    if args.list:
        list_comments()
        return

    # If --set is specified, set the comment
    if args.set:
        set_comment(args.set, env_name=args.env)
        return

    # If --append is specified, append to the comment
    if args.append:
        set_comment(args.append, env_name=args.env, append=True)
        # Show the updated comment after appending
        env_name = args.env or get_current_conda_env()
        hostname = get_hostname()
        updated_comment = get_comment(env_name, hostname)
        print(f"Updated comment: [{hostname}] {env_name}: {updated_comment}")
        return

    # No arguments, display the current environment comment
    env_name = args.env or get_current_conda_env()
    hostname = get_hostname()
    comment = get_comment(env_name, hostname)

    if comment:
        print(f"[{hostname}] {env_name}: {comment}")
    else:
        print(
            f"No comment found for environment [{env_name}] on host [{hostname}].")


if __name__ == "__main__":
    main()
