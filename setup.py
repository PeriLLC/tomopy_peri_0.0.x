#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
#    Copyright 2014-2015 Dake Feng, dakefeng@gmail.com
#
#    This file is part of TomograPeri.
#
#    TomograPeri is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    TomograPeri is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with TomograPeri.  If not, see <http://www.gnu.org/licenses/>.

#from distutils.core import setup, Extension
from setuptools import setup, Extension, find_packages

from distutils.command.build_ext import build_ext
from distutils.command.clean import clean
import os
import platform
import re
import subprocess
import sys
import zlib

#from setuptools import find_packages


VERSION = '0.0.1' 

NVIDIA_INC_DIRS = []
NVCC = 'nvcc'
for path in ('/usr/local/cuda', '/opt/cuda'):
    if os.path.exists(path):
        NVIDIA_INC_DIRS.append(os.path.join(path, 'include'))
        NVCC = os.path.join(path, 'bin', 'nvcc')
        break
else:
    print >>sys.stderr, "The CUDA compiler and headers required to build " \
                        "kernel were not found. Trying to continue anyway..."

EXTRA_COMPILE_ARGS = ['-Wall', '-fno-strict-aliasing', \
                      '-DVERSION="%s"' % (VERSION,)]


class GPUBuilder(build_ext):

    def _call(self, comm):
        p = subprocess.Popen(comm, stdout=subprocess.PIPE, shell=True)
        stdo, stde = p.communicate()
        if p.returncode == 0:
            return stdo
        else:
            print >>sys.stderr, "%s\nFailed to execute command '%s'" % \
                                (stde, comm)
            return None

    def _makedirs(self, pathname):
        try:
            os.makedirs(pathname)
        except OSError:
            pass

    def run(self):
        nvcc_o = self._call(NVCC + ' -V')
        if nvcc_o is not None:
            nvcc_version = nvcc_o.split('release ')[-1].strip()
        else:
            raise SystemError("Nvidia's CUDA-compiler 'nvcc' can't be " \
                          "found.")
        print "Compiling CUDA module using nvcc %s..." % nvcc_version

        bits, linkage = platform.architecture()
        if bits == '32bit':
            bit_flag = ' -m32'
        elif bits == '64bit':
            bit_flag = ' -m64'
        else:
            print >>sys.stderr, "Can't detect platform, using 32bit"
            bit_flag = ' -m32'

        nvcc_cmd = NVCC + bit_flag + ' -c -arch=sm_20 '\
                                 ' ./src/pml_cuda_kernel.cu' \
                                 ' --compiler-options ''-fPIC'''
        print "Executing '%s'" % nvcc_cmd
        subprocess.check_call(nvcc_cmd, shell=True)

        print "Building modules..."
        build_ext.run(self)


class GPUCleaner(clean):

    def _unlink(self, node):
        try:
            if os.path.isdir(node):
                os.rmdir(node)
            else:
                os.unlink(node)
        except OSError:
            pass

    def run(self):
        print "Removing temporary files and pre-built GPU-kernels..."
        try:
            for f in ('pml_cuda_kernel.o'):
                self._unlink(f)
        except Exception, (errno, sterrno):
            print >>sys.stderr, "Exception while cleaning temporary " \
                                "files ('%s')" % sterrno
        clean.run(self)


cuda_extension = Extension('lib.pml_cuda',
                    libraries = ['cuda', 'z'],
                    extra_objects = ['pml_cuda_kernel.o'],
                    sources = ['./src/pml_cuda.c'],
                    include_dirs = NVIDIA_INC_DIRS,
                    extra_compile_args = EXTRA_COMPILE_ARGS)

setup_args = dict(
        name = 'tomopy_peri',
        version = VERSION,
        packages=find_packages(),
        include_package_data=True,
        description = 'GPU-accelerated tomopy algorithms',
        long_description = \
            "Hardware accelleration for tomopy reconstruction algorithms. " \
            "The first version on pml algorithm.",
        license = 'GNU Less General Public License v3',
        author = 'Dake Feng',
        author_email = 'dakefeng@gmail.com',
        url = 'http://www.perillc.com',
        classifiers = \
              ['Development Status :: 4 - Beta',
               'Environment :: Console',
               'License :: OSI Approved :: GNU Less General Public License (LGPL)',
               'Natural Language :: English',
               'Operating System :: OS Independent',
               'Programming Language :: Python',
               'Topic :: Security'],
        platforms = ['any'],
        ext_modules = [cuda_extension],
        cmdclass = {'build_ext': GPUBuilder, 'clean': GPUCleaner},
        options = {'install': {'optimize': 1}, \
                    'bdist_rpm': {'requires': 'tomopy = 0.0.3'}})

if __name__ == "__main__":
    setup(**setup_args)

