# SDHR module implementation

The proposed SDHR-related modules are implemented in the modified Ultralytics YOLO files.

Main files:
- `conv.py`: contains the implementation of the SDHR-related module used in the model YAML.
- `__init__.py`: registers the custom module.
- `tasks.py`: imports and parses the custom module in the Ultralytics model parser.

To reproduce the model, copy these files to the corresponding locations in an Ultralytics YOLOv8 project, or manually merge the SDHR-related module definitions and registrations.
