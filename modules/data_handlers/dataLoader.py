import h5py
import ast
import pandas as pd
from pathlib import Path
import numpy as np
from scipy.signal import hilbert

from data_handlers.pwData import PWData, IQData, RFData

class DataLoader():
    def __init__(self, data_path:str, csv_path:str):
        """
        Load ultrasound PW dataset from PICMUS, CUBDL or custom data as PWData objects for reconstruction.

        Args:
            data_path (str): Filesystem path to the directory with PICMUS or CUBDL HDF5 files.
            df_path (str): Filesystem path to dataset csv.
        """
        
        self.path = Path(data_path)

        self.df = pd.read_csv(csv_path)
        self.df["path"] = None

        # === Mapping the names of datasets with their filesystem path in the df ===
        mapping: dict[str, str] = {}
        dataset_names = set(self.df["name"])

        for f in self.path.rglob('*.hdf5'):
            stem = f.stem
            if stem in dataset_names:
                mapping[stem] = str(f)

        self.df["path"] = self.df["name"].map(mapping.get)


    def get_defined_pwdata(self, selected_name: str, mode: str) -> PWData:
        df = self.df.query("name == @selected_name").iloc[0]

        name = df["name"]
        source = df["source"]

        params = {}
        params["name"] = df["name"]
        params["source"] = df["source"]
        params["pitch"] = df["pitch"] / 1000 # [mm] -> [m]
        params["angles_range"] = np.array(ast.literal_eval(df["angles_range"]))
        params["zlims"] = np.array(ast.literal_eval(df["zlims"]))
        params["fc"] = df["center_frecuency"] * 1e6 # [MHz] -> [Hz]
        

        path = Path(df["path"])
        with h5py.File(path, mode="r") as f:
            if source == "PICMUS":
                f = f["US"]["US_DATASET0000"]

            #for key in list(f.keys()):
                #print(f[key])

            params["c0"] = np.array(f["sound_speed"]).item()
            params["fdemod"] = np.array(f["modulation_frequency"]).item()

            if name[0:3] == "UFL":
                params["fs"] = np.array(f["channel_data_sampling_frequency"]).item()
            else:
                params["fs"] = np.array(f["sampling_frequency"]).item()     


            aperture_width = params["pitch"] * df["n_channels"]
            rad_range = np.radians(params["angles_range"])
            angles = np.linspace(rad_range[0], rad_range[1], num=df["n_angles"], dtype=np.float32)

            if name[0:3] in ("TSH", "MYO", "EUT", "INS"):
                params["t0"] = ((aperture_width/2) / params["c0"]) * np.abs(np.sin(angles))

                if name[0:3] == "EUT":
                    params["t0"] += 10 / params["fs"]

            elif name[0:3]  == "UFL":
                params["t0"] = -1 * np.array(f["channel_data_t0"], dtype=np.float32)
            
                if params["t0"].size == 1:
                    params["t0"] = np.ones_like(angles) * params["t0"]

            elif name[0:3] == "OSL":
                params["t0"] = -1 * np.array(f["start_time"], dtype=np.float32)
                params["t0"] = np.transpose(params["t0"])
            
            elif name[0:3] == "JHU":
                params["t0"] = -1 * np.array(f["time_zero"], dtype=np.float32)
                #params["t0"] = np.array(f["time_zero"], dtype=np.float32)

                #params["t0"] = ((aperture_width/2) / params["c0"]) * np.abs(np.sin(angles))
                #params["t0"] -= 10 / params["fs"]

            else:
                params["t0"] = np.zeros(df["n_angles"])
            
            

            if source == "CUBDL":
                data = np.array(f["channel_data"], dtype="float32")
            elif source == "PICMUS":
                data = np.array(f["data"]["real"], dtype="float32")
            else:
                raise ValueError("Source options are: 'CUBDL' and 'PICMUS'")
            
            if name[0:3] == "TSH":
                data = np.reshape(data, (128, df["n_angles"], -1))
                data = np.transpose(data, (1, 0, 2))
            

            if mode == "RF":
                params["rfdata"] = data
            else:
                if source == "CUBDL":
                    params["iqdata"] = np.stack((data, np.imag(hilbert(data, axis=-1))), axis=-1)
                    params["fdemod"] = 0
                elif source == "PICMUS":
                    params["iqdata"] = np.stack((data, np.array(f["data"]["imag"])), axis=-1)
                else:
                    raise ValueError("Source options are: 'CUBDL' and 'PICMUS'")

            params["n_angles"], params["n_channels"], params["n_samples"] = data.shape

            if mode == "RF":
                pw = RFData(**params)
            else:
                pw = IQData(**params)

        return pw
        
    #def get_custom_pwdata(self, y todos los parametros necesarios):

    #TODO: Funciones auxiliares para obtener por source y otros metodos.

    def get_df(self) -> pd.DataFrame:
        return self.df
