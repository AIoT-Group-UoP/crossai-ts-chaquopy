import pandas as pd

from caits.dataset import Dataset

data_dict = {
    "X": [pd.DataFrame(data=[1, 2, 3], columns=["Channel_1"])],
    "y": ["instance_label_name"],
    "id": ["instance_filename"],
}

dataset = Dataset(
    X=data_dict["X"],
    y=data_dict["y"],
    id=data_dict["id"],
)
