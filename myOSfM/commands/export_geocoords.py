# # pyre-strict
# import argparse

# from myOSfM.actions import export_geocoords
# from myOSfM.dataset import DataSet

# from . import command


# class Command(command.CommandBase):
#     name = "export_geocoords"
#     help = "Export reconstructions in geographic coordinates"

#     def run_impl(self, dataset: DataSet, args: argparse.Namespace) -> None:
#         export_geocoords.run_dataset(
#             dataset,
#             args.proj,
#             args.transformation,
#             args.image_positions,
#             args.reconstruction,
#             args.dense,
#             args.output,
#         )

#     def add_arguments_impl(self, parser: argparse.ArgumentParser) -> None:
#         parser.add_argument("--proj", help="PROJ.4 projection string", required=True)
#         parser.add_argument(
#             "--transformation",
#             help="Print cooordinate transformation matrix",
#             action="store_true",
#             default=False,
#         )
#         parser.add_argument(
#             "--image-positions",
#             help="Export image positions",
#             action="store_true",
#             default=False,
#         )
#         parser.add_argument(
#             "--reconstruction",
#             help="Export reconstruction.json",
#             action="store_true",
#             default=False,
#         )
#         parser.add_argument(
#             "--dense",
#             help="Export dense point cloud (depthmaps/merged.ply)",
#             action="store_true",
#             default=False,
#         )
#         parser.add_argument(
#             "--output", help="Path of the output file relative to the dataset"
#         )


# pyre-strict
import argparse
from myOSfM.actions import export_geocoords
from myOSfM.dataset import DataSet
from . import command

class Command(command.CommandBase):
    name = "export_geocoords"
    help = "Export reconstructions in geographic coordinates"

    def run_impl(self, dataset: DataSet, args: argparse.Namespace) -> None:
        export_geocoords.run_dataset(
            dataset,
            args.proj,
            args.transformation,
            args.image_positions,
            args.reconstruction,
            args.dense,
            args.output,
        )

    def add_arguments_impl(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--proj", help="PROJ.4 projection string", required=True)
        parser.add_argument(
            "--transformation",
            help="Print cooordinate transformation matrix",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "--image-positions",
            help="Export image positions",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "--reconstruction",
            help="Export reconstruction.json",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "--dense",
            help="Export dense point cloud (depthmaps/merged.ply)",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "--output", help="Path of the output file relative to the dataset"
        )
        parser.add_argument(
            "--offset-x",
            type=float,
            default=0.0,
            help="X offset for exported coordinates",
        )
        parser.add_argument(
            "--offset-y",
            type=float,
            default=0.0,
            help="Y offset for exported coordinates",
        )