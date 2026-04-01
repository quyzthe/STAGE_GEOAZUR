# # pyre-strict
# import argparse
# from types import ModuleType
# from typing import Callable, ContextManager, List

# from myOSfM import log
# from myOSfM.dataset import DataSet


# def command_runner(
#     all_commands_types: List[ModuleType],
#     dataset_factory: Callable[[str, str], ContextManager[DataSet]],
#     dataset_choices: List[str],
#     default_dataset_type: str = "opensfm",
# ) -> None:
#     """Main entry point for running the passed SfM commands types."""
#     log.setup()
#     # Create the top-level parser
#     parser = argparse.ArgumentParser()
#     subparsers = parser.add_subparsers(
#         help="Command to run", dest="command", metavar="command"
#     )


#     command_objects = [c.Command() for c in all_commands_types]


#     for command in command_objects:
#         subparser = subparsers.add_parser(command.name, help=command.help)
#         command.add_arguments(subparser)
#         subparser.add_argument(
#             "--dataset-type",
#             type=str,
#             required=False,
#             default=default_dataset_type,
#             choices=dataset_choices,
#         )

#     # Parse arguments
#     args = parser.parse_args()

#     # Instanciate datast
#     with dataset_factory(args.dataset, args.dataset_type) as data:
#         # Run the selected subcommand
#         for command in command_objects:
#             if args.command == command.name:
#                 command.run(data, args)


import argparse
import sys
import traceback
from types import ModuleType
from typing import Callable, ContextManager, List

from myOSfM import log
from myOSfM.dataset import DataSet

# for debug only
import inspect

def command_runner(
    all_commands_types: List[ModuleType],
    dataset_factory: Callable[[str, str], ContextManager[DataSet]],
    dataset_choices: List[str],
    default_dataset_type: str = "opensfm",
) -> None:
    log.setup()

    print("\n[DEBUG] Starting command_runner", flush=True)

    # Create parser
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        help="Command to run", dest="command", metavar="command"
    )

    command_objects = [c.Command() for c in all_commands_types]


    # Register commands
    for command in command_objects:
        # print(f"[DEBUG] Registering command: {command.name}", flush=True)

        subparser = subparsers.add_parser(command.name, help=command.help)
        command.add_arguments(subparser)

        subparser.add_argument(
            "--dataset-type",
            type=str,
            required=False,
            default=default_dataset_type,
            choices=dataset_choices,
        )

    # Parse arguments
    print("[DEBUG] Parsing arguments...", flush=True)
    args = parser.parse_args()

    print(f"[DEBUG] Parsed args: {vars(args)}", flush=True)

    if args.command is None:
        print("[ERROR] No command provided!", flush=True)
        parser.print_help()
        sys.exit(1)

    # Create command map (faster + clearer)
    command_map = {c.name: c for c in command_objects}

    if args.command not in command_map:
        print(f"[ERROR] Unknown command: {args.command}", flush=True)
        sys.exit(1)

    selected_command = command_map[args.command]

    print(f"[DEBUG] Selected command: {selected_command.name}", flush=True)
    print("[DEBUG] Command class:", type(selected_command))
    print("[DEBUG] Module:", selected_command.__class__.__module__)

    print("[DEBUG] File:", inspect.getfile(selected_command.__class__))
    print("[DEBUG] Source:")
    print(inspect.getsource(selected_command.run))


    # Instantiate dataset
    try:
        print(
            f"[DEBUG] Creating dataset: path={args.dataset}, type={args.dataset_type}",
            flush=True,
        )

        with dataset_factory(args.dataset, args.dataset_type) as data:
            print("[DEBUG] Dataset created successfully", flush=True)

            print("[DEBUG] Running command...", flush=True)
            selected_command.run(data, args)

            print("[DEBUG] Command finished successfully", flush=True)

    except Exception as e:
        print("\n[ERROR] Exception occurred!", flush=True)
        traceback.print_exc()
        sys.exit(1)

    finally:
        print("[DEBUG] command_runner finished", flush=True)