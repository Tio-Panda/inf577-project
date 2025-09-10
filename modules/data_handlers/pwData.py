import numpy as np
from abc import ABC

class PWData(ABC):
    skip_fields = ("angles", "probe_geometry", "t0")

    def __init__(
        self, 
        name, 
        source,
        pitch, 
        angles_range, 
        n_angles, 
        n_channels, 
        n_samples, 
        c0, 
        fs, 
        fdemod,
        t0,
        zlims,
        fc
    ):
        self.name = name
        self.source = source
        self.pitch = pitch
        self.angles_range = angles_range
        self.n_angles = n_angles
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.c0 = c0
        self.fs = fs
        self.fdemod = fdemod
        self.t0 = t0
        self.zlims = zlims
        self.fc = fc

        rad_range = np.radians(angles_range)
        self.angles = np.linspace(rad_range[0], rad_range[1], num=n_angles, dtype=np.float32)

        self.aperture_width = pitch * n_channels
        #x_pos = np.linspace(-self.aperture_width/2, self.aperture_width/2, self.n_channels)
        #self.probe_geometry = np.stack((x_pos, x_pos*0, x_pos*0), axis=1, dtype=np.float32)

        xpos = np.arange(n_channels) * pitch
        xpos -= np.mean(xpos)
        self.probe_geometry = np.stack([xpos, 0 * xpos, 0 * xpos], axis=1)

        last_sample_time = n_samples / fs
        self.img_depth = c0 / 2 * last_sample_time
        
    def __str__(self):
        units = {
            "fs": "Hz",
            "fdemod": "Hz",
            "pitch": "m",
            "aperture_width": "m",
            "img_depth": "m",
            "c0": "m/s"
        }
        lines = []

        for k, v in vars(self).items():
            if k in self.skip_fields:
                continue

            unit = f" {units[k]}" if k in units else ""
            lines.append(f"{k}: {v}{unit}")
        return "\n".join(lines)
    

class IQData(PWData):
    skip_fields = PWData.skip_fields + ("iqdata", )

    def __init__(self, iqdata, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iqdata = iqdata


class RFData(PWData):
    skip_fields = PWData.skip_fields + ("rfdata", )

    def __init__(self, rfdata, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rfdata = rfdata
