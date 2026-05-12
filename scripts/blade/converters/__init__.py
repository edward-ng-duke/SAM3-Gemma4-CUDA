"""Dataset converter scripts for the blade defect evaluation pipeline.

Each module in this package converts a third-party blade defect dataset
into the Roboflow YOLO layout consumed by ``RoboflowYoloLoader``:

    <out>/
        images/<stem>.jpg
        labels/<stem>.txt   # YOLO: ``cls cx cy w h`` (normalised)
        data.yaml           # ``nc`` + ``names``

These scripts are run manually (or via Makefile recipes) and live outside
the import path of the runtime evaluation code.
"""
