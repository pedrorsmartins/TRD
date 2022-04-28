#!/usr/bin/env python3

import os
import platform
import sys

from datetime import datetime
from setuptools import setup


SUPPORTED_PLATFORMS = [
    "linux",
    "win32",
]

__package_name__ = "nvidia-cuda-runtime-cu11"
__description__  = "CUDA Runtime native Libraries Metapackage"
__version__      = datetime.now().strftime("%Y.%m.%d")


if __name__ == "__main__":

    # ┍━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━┑
    # │ System              │ `sys.platform` Value │
    # ┝━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━┥
    # │ Linux               │ linux                │
    # │ Windows             │ win32                │
    # │ Windows/Cygwin      │ cygwin               │
    # │ Windows/MSYS2       │ msys                 │
    # │ Mac OS X            │ darwin               │
    # │ OS/2                │ os2                  │
    # │ OS/2 EMX            │ os2emx               │
    # │ RiscOS              │ riscos               │
    # │ AtheOS              │ atheos               │
    # │ FreeBSD 7           │ freebsd7             │
    # │ FreeBSD 8           │ freebsd8             │
    # │ FreeBSD N           │ freebsdN             │
    # │ OpenBSD 6           │ openbsd6             │
    # ┕━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━┙
    if sys.platform not in SUPPORTED_PLATFORMS:
        raise OSError("Your operating system is not supported by "
                      "`{_pkg}`: {_platform}.\nOnly these systems are "
                      "supported: `{_supported}`.".format(
                        _pkg=__package_name__,
                        _platform=sys.platform,
                        _supported=SUPPORTED_PLATFORMS
                      )
        )

    if platform.machine().lower() not in ["x86_64", "amd64"]:
        raise OSError("Your CPU architecture is not supported yet. Only x86_64 "
                      "on Linux and AMD64 on windows are supported. Received: "
                      "{}".format(platform.machine()))

    install_requires = []
    if "egg_info" in sys.argv:
        import _pynvml as pynvml
        pynvml.nvmlInit()

        try:
            __driver_version__ = str(pynvml.nvmlSystemGetCudaDriverVersion_v2())
        except:
            __driver_version__ = str(pynvml.nvmlSystemGetCudaDriverVersion())

        if __driver_version__[:4] == "1104":
            install_requires.append(__package_name__+"4")

        pinned_version = os.environ.get("NVIDIA_VER_PINNING", None)
        if pinned_version is not None:
            if (
                pinned_version[:2] not in ["==", ">=", "<="] and
                pinned_version[0] not in [">", "<"]
            ):
                raise ValueError(
                    "NVIDIA_VER_PINNING must start by on of: [==, >=, <=, <, >]"
                )
            install_requires[0] += "{}".format(pinned_version)

        print("\n########################################")
        print("CUDA Driver Version:", __driver_version__)
        print("Dependency:", install_requires)
        print("########################################\n")


    setup(
        name=__package_name__,
        version=__version__,
        description=__description__,
        url="https://developer.nvidia.com/cuda-zone",
        author="Nvidia CUDA Installer Team",
        author_email="cuda_installer@nvidia.com",
        license="NVIDIA Proprietary Software",
        install_requires=install_requires,
        include_package_data=True,
    )