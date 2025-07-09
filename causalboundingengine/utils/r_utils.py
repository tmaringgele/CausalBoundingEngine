def ensure_r_ready(r_path: str = None):
    """
    Ensure that R and rpy2 are available and that R_HOME is configured.
    Optionally allow manual R path override via `r_path`.
    """
    import os
    import subprocess

    if r_path:
        os.environ['R_HOME'] = r_path

    if 'R_HOME' not in os.environ:
        try:
            r_home = subprocess.check_output(["R", "RHOME"], text=True).strip()
            os.environ['R_HOME'] = r_home
        except Exception as e:
            raise RuntimeError(
                "R is not installed or not found in PATH. "
                "Please install R and make sure it's accessible from the command line."
            ) from e

    try:
        import rpy2.robjects as robjects
        import rpy2.robjects.packages as rpackages
        import rpy2.robjects.vectors as rvectors
    except ImportError as e:
        raise ImportError(
            "CausalOptim requires the optional Python package 'rpy2'.\n"
            "Install it with: pip install causalboundingengine[r]"
        ) from e

    if not rpackages.isinstalled("causaloptim"):
        print("Installing R package 'causaloptim'...")

        utils = rpackages.importr("utils")
        utils.chooseCRANmirror(ind=1)
        utils.install_packages(rvectors.StrVector(["causaloptim"]))
