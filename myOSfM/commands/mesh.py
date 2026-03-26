# pyre-strict
import argparse

from myOSfM.actions import mesh
from myOSfM.dataset import DataSet

from . import command


class Command(command.CommandBase):
    name = "mesh"
    help = "Add delaunay meshes to the reconstruction"

    def run_impl(self, dataset: DataSet, args: argparse.Namespace) -> None:
        mesh.run_dataset(dataset)

    def add_arguments_impl(self, parser: argparse.ArgumentParser) -> None:
        pass
