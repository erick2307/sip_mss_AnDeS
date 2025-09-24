import glob
import os
import time
import pickle
import cloudpickle
from datetime import datetime, date, timedelta

import cv2
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # to solve issue on out of memory
import contextily as ctx
from cmcrameri import cm
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import zipfile
import report as log

plt.style.use("default")

# TODO: add a function to store data in tiff format
# FIXME: x-axis of plots are for interval times not a specific time.

def read_object(path):
    """To load data from a pickle"""
    with open(Path(path), "rb") as handle:
        obj = pickle.load(handle)
    return obj


def update_server():
    """
    To update and sync AWS and Pegasus32

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    cwd = os.getcwd()
    server = "/Volumes/Pegasus32/data/AWS_NTT_Data"
    os.chdir(server)
    command = "aws s3 sync s3://client-rt2-01h-6-tohokuuniversity2023/realtime ."
    os.system(command=command)
    os.chdir(cwd)
    print("Data downloaded")
    return


# search for the ZIP files on a specific year.month.date
def zippedfiles(year=2022, month=10, date=1):
    folder = f"/Volumes/Pegasus32/data/AWS_NTT_Data/{year}{month:02d}{date:02d}"
    zipfiles = sorted(glob.glob(os.path.join(folder, "*.zip")))
    return zipfiles


# extract all zippedfiles into a new folder
def extractfiles(year=2019, month=1, date=1):
    datefolder = f"{year}{month:02d}{date:02d}"
    if not os.path.exists(
        os.path.join("/Volumes/Pegasus32/data/AWS_NTT_Data", datefolder)
    ):
        return f"/Volumes/Pegasus32/data/AWS_NTT_Data/{datefolder} does not exist"
    outfolder = f"/Volumes/Pegasus32/data/NTT_Data/{year}_csv/{datefolder}"
    if not os.path.exists(outfolder):
        os.mkdir(outfolder)
    zipfiles = zippedfiles(year, month, date)
    # extract allfiles
    for f in tqdm(zipfiles, desc=f"{datefolder}", position=1):
        with zipfile.ZipFile(f, "r") as zip_ref:
            zip_ref.extractall(f"{outfolder}")
    return


def unzip(start, end):
    dates = pd.date_range(start, end)
    print(f"Extracting {len(dates)} days ...")
    file_log = tqdm(total=0, position=2, bar_format="{desc}")
    for date in tqdm(dates, desc="Dates", position=0):
        s = time.time()
        extractfiles(date.year, date.month, date.day)
        t1 = time.time() - s
        file_log.set_description_str(f"Folder:{date}, Loadtime:{t1}s")
    return


class MobileData:
    """
    A class representing mobile data.

    Parameters:
    - one_mesh (bool): Flag indicating whether to use one mesh or not.
    - aoi_name (str): Name of the area of interest.
    - aoi_pol (str): Path to the AOI_POLYGON file.
    - aoi_mesh (str): Path to the AOI_MESH file.
    - event_name (str): Name of the event.
    - dt_start (str): Start date of the data.
    - dt_main (str): Main date of the data.
    - dt_end (str): End date of the data.
    - fpfx (str): File prefix.

    Attributes:
    - dfd (dict): Dataframe of one day data (dynamic).
    - gdfp (dict): GeoDataframe of period data (fix).
    - dft (dict): Dataframe temporal (when reading from pickle).
    - bbox (set): Set of bounding boxes.
    - pop (DataFrame): Population data.
    - folderpath (Path): Path to the folder.
    - one_mesh (bool): Flag indicating whether to use one mesh or not.
    - aoi_name (str): Name of the area of interest.
    - aoi_pol (GeoDataFrame): AOI_POLYGON data.
    - aoi_mesh (GeoDataFrame): AOI_MESH data.
    - e_name (str): Event name.
    - sdate (Timestamp): Start date of the data.
    - mdate (Timestamp): Main date of the data.
    - edate (Timestamp): End date of the data.
    - date_idx (DatetimeIndex): Date index.
    - hour_idx (DatetimeIndex): Hour index.

    Methods:
    - _create_directory(path): Creates a directory.
    - read_one_day_data(y, m, d, ftype): Reads one day data.
    - read_period_data(ftype): Reads period data.
    - get_population(ftype): Calculates population in period.
    - _store_dict(dfd, name): Stores data in a pickle.
    - _store_object(name): Stores class object in a pickle.
    - _store_object_cloudpickle(name): Stores class object in a pickle using cloudpickle.
    - read_dict(path, name): Loads data from a pickle.
    - create_date_idx(start, end, freq): Creates a set date index.
    - read_mesh(path): Reads a mesh data.
    - intersect_data_mesh(df, mesh): Intersects data with mesh.
    - create_dict_area(dfd, date_idx, n_hours, area, save, name): Creates a dictionary of areas.
    - create_array_one_meshgrid(dfd, meshcode, date_idx, n_hours, save, name): Creates an array for one meshgrid.
    """
    def __init__(
        self,
        one_mesh,
        aoi_name,
        aoi_pol,
        aoi_mesh,
        event_name,
        dt_start,
        dt_main,
        dt_end,
        fpfx,
    ):
        self.dfd = {}  # Dataframe of one day data (dynamic)
        self.gdfp = {}  # GeoDataframe of period data (fix)
        self.dft = {}  # Dataframe temporal (when reading from pickle)
        self.bbox = set()
        self.pop = pd.DataFrame()
        self.fpfx = fpfx
        self.one_mesh = one_mesh
        self.aoi_name = aoi_name
        self.folderpath = Path(
            self.aoi_name, str(datetime.now().strftime("%Y%m%d%H%M%S"))
        )
        self._create_directory(self.folderpath)
        self._create_directory(Path(self.folderpath, "plots"))
        self._create_directory(Path(self.folderpath, "figures"))
        self._create_directory(Path(self.folderpath, "data"))
        if aoi_pol == None and aoi_mesh == None:
            log.log_error("No file found. Provide AOI_POLYGON or AOI_MESH")
            raise OSError("No file found. Provide AOI_POLYGON or AOI_MESH")
        if aoi_pol != None:
            self.aoi_pol = gpd.read_file(aoi_pol)
        else:
            log.log_info("No AOI_POLYGON path provided. Trying with AOI_MESH...")
            # print("No AOI_POLYGON path provided. Trying with AOI_MESH...")
        if aoi_mesh == None:
            log.log_info("Creating mesh file...")
            # print("Creating mesh file...")
            self.aoi_mesh = gpd.read_file(
                "/Volumes/Pegasus32/japan/mesh/japan_mesh4_CRS84.geojson",
                mask=self.aoi_pol,
            )
        else:
            log.log_info("Reading mesh from file...")
            # print("Reading mesh from file...")
            self.aoi_mesh = gpd.read_file(aoi_mesh)
        self.aoi_mesh.to_file(
            Path(
                self.folderpath, "data", f"./{self.fpfx}_{self.aoi_name}_mesh.geojson"
            ),
            driver="GeoJSON",
        )
        log.log_info("AOI_MESH file saved in 'data' folder.")
        # print("AOI_MESH file saved in 'data' folder.")
        self.e_name = event_name
        self.sdate = pd.to_datetime(dt_start)
        if dt_main == None:
            self.mdate = pd.to_datetime(dt_start)
            log.log_info("Main date set to same as Start date.")
            # print("Main date set to same as Start date.")
        else:
            self.mdate = pd.to_datetime(dt_main)
        self.edate = pd.to_datetime(dt_end)
        self.date_idx = pd.date_range(self.sdate, self.edate, freq="d")
        self.hour_idx = pd.date_range(self.sdate, self.edate, freq="H")
        log.log_blank_line()
        log.log_separator()
        log.log_info(
            f"""
              ===== {self.e_name} =====
              1. Event Name     ==> {self.e_name}
              2. Folder         ==> {self.aoi_name}
              3. Working CRS    ==> {self.aoi_mesh.crs.to_string()} 
              4. One mesh       ==> {self.one_mesh}
              5. Start Date     ==> {self.sdate}
              6. Main Date      ==> {self.mdate}
              7. End Date       ==> {self.edate}
              ========================
              """
        )
        return

    def _create_directory(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        print(f"Folder {path} created")
        return

    def read_one_day_data(self, y=2019, m=1, d=1, ftype=0):
        """Read one day data and
        return a dictionary
        of the day key: hour"""
        outfolder = f"/Volumes/Pegasus32/data/NTT_Data/{y}_csv/{y}{m:02d}{d:02d}"
        # create csv path and filenames list
        csvfiles = sorted(glob.glob(os.path.join(outfolder, f"*{ftype}.csv")))
        # read all data
        self.dfd = {k: pd.read_csv(c) for k, c in enumerate(csvfiles)}
        return

    #! This function takes long -> maybe because I was running in office pc reading from mstudio via network
    def read_period_data(self, ftype=0):
        for date in self.date_idx:
            y = date.date().year
            m = date.date().month
            d = date.date().day
            self.read_one_day_data(y, m, d, ftype)
            for hour in self.dfd.keys():
                gdf = self.intersect_data_mesh(self.dfd[hour], self.aoi_mesh)
                self.gdfp[f"{y}{m:02d}{d:02d}{hour:02d}00"] = gdf
        # might be not so efficient doing it here
        self._max_min_pop_period()
        self._store_dict(self.gdfp, name=f"gdf_dict_ftype{ftype}")
        return

    def get_population(self, ftype=0):
        if self.gdfp == {}:
            print("Reading period data...")
            self.read_period_data(ftype)
        # self.pop = []
        print("Calculating population in period...")
        # for i in range(len(self.gdfp)):
        #     self.pop.append(self.gdfp[list(self.gdfp.keys())[i]]["population"].sum())
        pop_dict = {}
        for i, key in enumerate(self.gdfp.keys()):
            df = self.gdfp[key]
            pop_dict[key] = df["population"].sum()
            self.pop = pd.DataFrame.from_dict(
                pop_dict, orient="index", columns=["population"]
            )
            # Converting the index as date
            self.pop.index = pd.to_datetime(self.pop.index)
        self._store_object(name=f"nttclass_ftype{ftype}")
        return

    def _store_dict(self, dfd, name="data"):
        """To store data in a pickle"""
        with open(
            Path(self.folderpath, "data", f"{self.fpfx}_{name}.pickle"), "wb"
        ) as handle:
            pickle.dump(dfd, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return

    def _store_object(self, name="nttclass_ftype0"):
        """To class object in a pickle"""
        with open(
            Path(self.folderpath, "data", f"{self.fpfx}_{name}.pickle"), "wb"
        ) as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return

    def _store_object_cloudpickle(self, name="nttclass_ftype0"):
        """To class object in a pickle"""
        with open(
            Path(self.folderpath, "data", f"{self.fpfx}_{name}.pickle"), "wb"
        ) as handle:
            cloudpickle.dump(self, handle)
        return

    def read_dict(self, path, name="data"):
        """To load data from a pickle"""
        with open(Path(path, f"{self.fpfx}_{name}.pickle"), "rb") as handle:
            self.dft = pickle.load(handle)
        return self.dft

    def create_date_idx(self, start="20190101", end="20200101", freq="d"):
        """To create a set date index"""
        sdate = date(int(start[:4]), int(start[4:6]), int(start[-2:]))  # start date
        edate = date(
            int(end[:4]), int(end[4:6]), int(end[-2:])
        )  # end date (not inclusive in this pandas version)
        if freq == "d":
            date_idx = (
                pd.date_range(sdate, edate - timedelta(days=1), freq=freq)
                .strftime("%Y%m%d")
                .to_list()
            )
        elif freq == "H":
            date_idx = (
                pd.date_range(sdate, edate, freq=freq)
                .strftime("%Y%m%d%H")
                .to_list()[:-1]
            )
        else:
            date_idx = None
            print('input "freq"')
        return date_idx

    def read_mesh(self, path):
        """reads a mesh data with index,centroid,x,y
        creates a DF and Points for geometry to return a
        """
        area = pd.read_csv(path, index_col=0)
        area = gpd.GeoDataFrame(
            area, geometry=gpd.points_from_xy(x=area["X"], y=area["Y"]), crs="EPSG:4326"
        )
        area.drop(columns=["centroid", "X", "Y"], inplace=True)
        return area

    def intersect_data_mesh(self, df, mesh):
        mesh["MESH4_ID"] = list(mesh["MESH4_ID"].values.astype("int64"))
        dfa = df[df["area"].isin(list(mesh["MESH4_ID"].values))]
        dfa = dfa.merge(mesh, left_on="area", right_on="MESH4_ID")
        # dfa.drop(columns=["MESH4_ID"], inplace=True)
        gdf = gpd.GeoDataFrame(dfa, geometry="geometry")
        return gdf

    def create_dict_area(
        self, dfd, date_idx, n_hours, area, save=True, name="dict_area"
    ):
        df = {}
        for key in date_idx:
            for h in range(n_hours):
                new_key = key + f"{h:02}"
                df[new_key] = self.intersect_data_mesh(dfd[key][h], area)
                if save:
                    self.store_dict(df, f"{name}")
        return df

    def create_array_one_meshgrid(
        self, dfd, meshcode, date_idx, n_hours, save=False, name="one_mesh"
    ):
        # In mesh : rows = days, columns = hours
        mesh = np.zeros((len(date_idx), n_hours), dtype=np.int64)
        # In mesh : rows = hours, columns = days
        mesh_t = np.zeros((n_hours, len(date_idx)), dtype=np.int64)
        for i, key in enumerate(date_idx):
            for h in range(n_hours):
                new_key = key + f"{h:02}"
                df = dfd[new_key]
                tempdf = df[df.area == meshcode]
                if tempdf.empty:
                    mesh[i][h] = 0
                    mesh_t[h][i] = 0
                else:
                    mesh[i][h] = tempdf["population"].to_list()[0]
                    mesh_t[h][i] = tempdf["population"].to_list()[0]
        if save:
            np.savetxt(
                Path(self.folderpath, "data", f"{self.fpfx}_{name}.csv"),
                mesh,
                delimiter=",",
            )
            np.savetxt(
                Path(self.folderpath, "data", f"{self.fpfx}_{name}_t.csv"),
                mesh_t,
                delimiter=",",
            )
        return mesh, mesh_t

    def meshgrid_array_to_dataframe(self, mesh, date_idx, n_hours=24):
        mesh_1 = pd.DataFrame(
            mesh, columns=np.linspace(0, n_hours - 1, n_hours).astype("int")
        )
        mesh_1.set_index(pd.Index(date_idx), inplace=True)
        mesh_1.index = pd.to_datetime(mesh_1.index)
        return mesh_1

    def bounding_box(self):
        # calculate boundaries
        bbox = np.zeros((len(self.gdfp.keys()), 4))
        for i, key in enumerate(self.gdfp.keys()):
            bbox[i, :] = self.gdfp[str(key)].total_bounds

        self.xmin = bbox[:, 0].min()
        self.ymin = bbox[:, 1].min()
        self.xmax = bbox[:, 2].max()
        self.ymax = bbox[:, 3].max()

        self.bbox = {self.xmin, self.xmax, self.ymin, self.ymax}
        return

    def _max_min_pop_period(self):
        _pmax = np.nan
        _pmin = np.nan
        self.pmax = 0
        self.pmin = 10**10
        for i, key in enumerate(self.gdfp.keys()):
            _pmax = self.gdfp[str(key)].population.max()
            _pmin = self.gdfp[str(key)].population.min()
        if _pmax > self.pmax:
            self.pmax = _pmax
        if _pmin < self.pmin:
            self.pmin = _pmin
        return

    def plot_population(self, ftype=0, save=True):
        if self.pop.empty:
            print("Getting population...")
            self.get_population(ftype)
        plt.close()
        fig, ax = plt.subplots(figsize=(20, 5))
        ax.plot(self.pop.population)
        # p_min_x = self.pop.index(min(self.pop))
        p_min_x = self.pop[["population"]].idxmin().item()
        # p_max_x = self.pop.index(max(self.pop))
        p_max_x = self.pop[["population"]].idxmax().item()
        p_min_y = self.pop.population.min()  # min(self.pop)
        p_max_y = self.pop.population.max()  # max(self.pop)
        ymin, ymax = ax.get_ylim()
        ax.vlines(self.mdate, ymin, ymax, colors="grey", linestyles="dotted")
        ax.scatter([p_min_x, p_max_x], [p_min_y, p_max_y], c="r")
        ax.annotate(
            p_min_y, (p_min_x, p_min_y), (p_min_x + timedelta(hours=0.2), p_min_y + 5)
        )
        ax.annotate(
            p_max_y, (p_max_x, p_max_y), (p_max_x + timedelta(hours=0.2), p_max_y + 5)
        )
        ax.annotate(
            self.mdate,
            (self.mdate, p_max_y),
            (self.mdate + timedelta(hours=0.2), p_max_y + 5),
        )
        ax.set_xlabel("Days")
        ax.set_ylabel("Population")
        ax.set_title(f"Aggregated Population at {self.e_name}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        # trans = ax.get_xaxis_transform()  # x in data untis, y in axes fraction
        # ann = ax.annotate(f"{self.sdate} ~ {self.edate}", xy=(0, -0.1), xycoords=trans)
        if save:
            plt.savefig(
                Path(self.folderpath, "plots", f"{self.fpfx}_pop_ftype{ftype}.png"),
                dpi=300,
            )
        return

    def plot_gdf(self, gdf, cmap="cividis_r", save=True):
        plt.close()
        fig, ax = plt.subplots(1, 1)
        gdf.plot(
            ax=ax,
            # aspect="equal",
            column="population",
            # marker="s",
            # markersize=45,
            cmap=cmap,
            vmin=self.pmin,
            vmax=self.pmax,  # maximum as Tokyo density 6,158 pers/km2 <> 1,500 pers./(500x500)m2
            # figsize=(10, 10),
            alpha=0.6,
            legend=True,
            legend_kwds={"label": "Population"},
        )
        ctx.add_basemap(
            ax,
            zoom=15,
            crs=gdf.crs.to_string(),
            source=ctx.providers.Esri.WorldImagery,
            attribution=False,
        )
        ax.ticklabel_format(useOffset=False, style="plain")
        plt.title(f'{gdf["date"][0]} {int(gdf["time"][0]/100):02d}:00')
        if len(self.bbox) == 0:
            self.bounding_box()
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        if save:
            plt.savefig(
                Path(
                    self.folderpath,
                    "figures",
                    f'{self.fpfx}_{gdf["date"][0]}{gdf["time"][0]:04}.png',
                ),
                dpi=300,
                bbox_inches="tight",
            )
        return

    def plot_aoi(self, gdf, save=False):
        plt.close()
        fig, ax = plt.subplots(1, 1)
        gdf.plot(
            ax=ax,
            column="population",
            alpha=0.0,
        )
        ctx.add_basemap(
            ax,
            zoom=15,
            crs=gdf.crs.to_string(),
            source=ctx.providers.Esri.WorldImagery,
            attribution=False,
        )
        ax.ticklabel_format(useOffset=False, style="plain")
        plt.title(f"{self.e_name}")
        if len(self.bbox) == 0:
            self.bounding_box()
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        if save:
            plt.savefig(
                Path(self.folderpath, "plots", f"{self.fpfx}_{self.aoi_name}.png"),
                dpi=300,
                bbox_inches="tight",
            )
        return

    def make_video_(
        self,
        image_foldername="figures",
        video_name="video",
        fps=1,
        verbose=False,
    ):
        image_folder = Path(self.folderpath, image_foldername)
        video_name = str(Path(self.folderpath, f"{video_name}.mp4"))
        images = [
            img for img in sorted(os.listdir(image_folder)) if img.endswith(".png")
        ]
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape
        video = cv2.VideoWriter(
            video_name,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        for image in images:
            if verbose:
                print(image)
            video.write(cv2.imread(os.path.join(image_folder, image)))
        cv2.destroyAllWindows()
        video.release()
        return

    # better this one?
    def make_video(self, framerate=5):
        cwd = os.getcwd()
        os.chdir(Path(self.folderpath, "figures"))
        os.system(
            f'ffmpeg -framerate {framerate} -pattern_type glob -i "*.png" -s 3840x2160 -pix_fmt yuv420p ../{self.fpfx}_map.mp4'
        )
        os.chdir(cwd)
        return
