from setuptools import setup
from distutils.command.install import install as DistutilsInstall
from setuptools.command.install import install
from distutils.command.build import build
import os
from multiprocessing import cpu_count
from subprocess import call

BASEPATH = os.path.dirname(os.path.abspath(__file__))
BTM_PATH = os.path.join(BASEPATH, 'btm')

class XCSoarBuild(build):
    def run(self):
        # run original build code
        build.run(self)

        # build XCSoar
        build_path = os.path.abspath(self.build_temp)

        cmd = ['make']

        try:
            cmd.append('-j%d' % cpu_count())
        except NotImplementedError:
            print('Unable to determine number of CPUs. Using single threaded make.')

        def compile():
            call(cmd, cwd=BTM_PATH)

        self.execute(compile, [], 'Compiling xcsoar')
        print('a')
        # copy resulting tool to library build folder
        self.mkpath(self.build_lib)
        target_files = [os.path.join(BTM_PATH, 'btm.so')]
        if not self.dry_run:
            for target in target_files:
                self.copy_file(target, self.build_lib)




setup(
    name='btm',
    version='0.1.0',
    packages=['btm'],
    cmdclass={
        'build': XCSoarBuild
    }
)

