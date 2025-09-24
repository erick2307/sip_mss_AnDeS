import pandas as pd
import numpy as np
import ntt_dev
import mossad
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import time as clock
from datetime import datetime, time, date, timedelta
from PIL import Image
from pathlib import Path
from tqdm import tqdm


def main():
    """
    Anomaly Detection System - AnDeS

    This script provides an interface to visualize NTT Docomo data and perform anomaly detection.

    Usage:
    1. Set the input parameters such as start date, end date, main event date, etc.
    2. Verify the data availability and download if necessary.
    3. Set the options for calculating and plotting.
    4. Get the data and perform anomaly detection if selected.
    5. View the outputs.

    Note: This is the developer version of the script.

    Args:
        None

    Returns:
        None
    """
    # layouts and containers
    # sidebar
    # add_sidebar = st.sidebar.selectbox("Select Period", ("Example 1", "Example 2"))

    # containers
    head = st.container()
    "---"
    inputs = st.container()
    "---"
    monitor = st.container()
    "---"
    outputs = st.container()

    # SECTIONS
    # HEAD
    head.title("Anomaly Detection System - AnDeS")
    head.title(":green[Developer Version]")
    # head.subheader("Data available locally at lab")
    # head.write("This is an interface to visualize NTT Docomo data")


    # INPUTS
    inputs.header("1. INPUTS")
    i_col1, i_col2 = inputs.columns(2)
    with i_col1:
        start_date = st.date_input(
            ":blue[Start Date]",
            value=date(2022, 3, 15),
            min_value=date(2016, 1, 1),
            max_value=date.today()# - timedelta(days=1),
        )
        start_time = st.time_input(
            ":blue[Start time]",
            value=time(0, 0),
            step=3600,
        )

    with i_col2:
        end_date = st.date_input(
            ":blue[End Date]",
            value=date(2022, 3, 17),
            min_value=date(2016, 1, 1),
            max_value=date.today()# - timedelta(days=1),
        )
        end_time = st.time_input(
            ":blue[End time]",
            value=time(23, 0),
            step=3600,
        )

    main_show_bool = inputs.checkbox("Is there a main event date/time to plot?")

    main_date = None
    main_time = None
    if main_show_bool:
        main_date = inputs.date_input(
            ":red[Main Date]",
            value=date(2022, 3, 16),
            min_value=date(2016, 1, 1),
            max_value=date.today()# - timedelta(days=1),
        )
        main_time = inputs.time_input(":red[Main time]", value=time(23, 36), step=60)

    MESH_ID = None
    ROOT_FOLDER_NAME = inputs.text_input(
        label="Root folder name",
        value="watari",
        help="Set a single name for the root folder. Avoid spaces.",
    )
    i_col3, i_col4 = inputs.columns(2)
    with i_col3:
        AOI_POLYGON = st.file_uploader("Option 1: Upload Polygon file", type=["geojson"])

    with i_col4:
        AOI_MESH = st.file_uploader("Option 2: Upload Mesh file", type=["geojson"])

    EVENT_NAME = inputs.text_input(
        label="Name of the event or project", value="2022 Fukushima EQ"
    )
    EVENT_DATE_START = datetime.combine(start_date, start_time)
    if main_show_bool:
        EVENT_DATE_MAIN = datetime.combine(main_date, main_time)
    else:
        EVENT_DATE_MAIN = None
    EVENT_DATE_END = datetime.combine(end_date, end_time)
    FILE_PREFIX = inputs.text_input(
        label="File Prefix",
        value="watari",
        help="Write a keyword for prefix in your output file names.",
    )
    FILE_TYPE = inputs.number_input(
        label="Input type of file",
        min_value=0,
        max_value=1,
        value=0,
        help="""\n
        0: Aggregated Population\n
        1: Population with ages\n
        2: Population with origin prefecture\n
        3: Population with origin city
        """,
    )

    # MONITOR
    down_hid_bool = True
    getdata_hid_bool = True
    monitor.header("2. VERIFY YOUR INPUTS")
    monitor.write(f"Start: {start_date}, {start_time}")
    monitor.write(f"End: {end_date}, {end_time}")
    monitor.write(f"Main Event: {main_date}, {main_time}")

    m_col11, m_col12 = monitor.columns(2)
    with m_col11:
        verifydata = st.button("Verify Data Availability")
    with m_col12:
        download = st.button("Download?", disabled=down_hid_bool)

    if verifydata:
        # check folders
        root = "/Volumes/Pegasus32/data/NTT_Data"
        date_chck = EVENT_DATE_START
        folder = (
            f"{date_chck.year}_csv/{date_chck.year}{date_chck.month:02d}{date_chck.day:02d}"
        )
        file = f"{date_chck.year}_csv/{date_chck.year}{date_chck.month:02d}{date_chck.day:02d}/clipped_mesh_pop_{date_chck.year}{date_chck.month:02d}{date_chck.day:02d}{date_chck.hour:02d}00_00000.csv"
        startfolder = Path(root, folder)
        startfile = Path(root, file)

        date_chck = EVENT_DATE_END
        folder = (
            f"{date_chck.year}_csv/{date_chck.year}{date_chck.month:02d}{date_chck.day:02d}"
        )
        file = f"{date_chck.year}_csv/{date_chck.year}{date_chck.month:02d}{date_chck.day:02d}/clipped_mesh_pop_{date_chck.year}{date_chck.month:02d}{date_chck.day:02d}{date_chck.hour:02d}00_00000.csv"
        endfolder = Path(root, folder)
        endfile = Path(root, file)

        if (
            not startfolder.exists()
            or not endfolder.exists()
            or not startfile.exists()
            or not endfile.exists()
        ):
            monitor.write(f"ATTENTION! Data has not been downloaded to the server yet!")
            down_bool = False
        else:
            monitor.write(f"DATA AVAILABLE!")
            # getdata_hid_bool = False

    if download:
        ntt_dev.update_server()
        ntt_dev.unzip(start=EVENT_DATE_START, end=EVENT_DATE_END)
        # getdata_hid_bool = False

    ##### MAIN PROGRAM #####
    monitor.write("Select options to calculate:")
    plot_gdf_bool = monitor.checkbox("Plot maps?", value=False)
    make_video_bool = monitor.checkbox("Make videos?", value=False)
    store_data_bool = monitor.checkbox("Store Data?", value=True)
    anomaly_detection_bool = monitor.checkbox("Get Anomaly Detection?", value=False)
    if anomaly_detection_bool:
        m_col21, m_col22 = monitor.columns(2)
        with m_col21:
            window_size = st.number_input("Window size in hours", min_value=3, value=3)
        with m_col22:
            n_anomalies = st.number_input("Number of anomalies to extract", value=1)

    set_options = monitor.button("Set Options")
    if set_options:
        getdata_hid_bool = False
    getdata = monitor.button("Get Data", disabled=getdata_hid_bool)
    if getdata:
        # Load data
        t0 = clock.time()
        with st.spinner("Calculating..."):
            case = ntt_dev.MobileData(
                one_mesh=MESH_ID,
                aoi_name=ROOT_FOLDER_NAME,
                aoi_pol=AOI_POLYGON,
                aoi_mesh=AOI_MESH,
                event_name=EVENT_NAME,
                dt_start=EVENT_DATE_START,
                dt_main=EVENT_DATE_MAIN,
                dt_end=EVENT_DATE_END,
                fpfx=FILE_PREFIX,
            )
            case.plot_population(ftype=FILE_TYPE, save=True)
            t1 = clock.time()
            monitor.write(f"Population plot: {t1 - t0:.2f}s")
            if store_data_bool:
                case._store_dict(case.gdfp)
                monitor.write("Data stored")
            if plot_gdf_bool:
                for key in tqdm(case.gdfp.keys()):
                    case.plot_gdf(case.gdfp[key], cmap="cividis_r", save=True)
                t2 = clock.time()
                monitor.write(f"GDFs plots: {t2 - t1:.2f}s")
            if make_video_bool:
                case.make_video()
                t3 = clock.time()
                monitor.write(f"Make video: {t3 - t2:.2f}s")
            if anomaly_detection_bool:
                case_an = mossad.Mossad(
                    case,
                    case.pop,
                    event_date=case.mdate,
                    m=window_size,
                    nd=n_anomalies,
                    save=True,
                    workspace=case.folderpath,
                )
                t4 = clock.time()
                if make_video_bool:
                    case_an.make_video()
                    monitor.write(f"Make video: {t4 - t3:.2f}s")
            monitor.write(f"Calculation time: {clock.time() - t0:.2f}s")

    # OUTPUTS
    outputs.header("3. OUTPUTS")
    if getdata:
        img = Image.open(
            Path(case.folderpath, "plots", f"{case.fpfx}_pop_ftype{FILE_TYPE}.png")
        )
        outputs.image(img)
        if make_video_bool:
            video_file = open(Path(case.folderpath, f"{case.fpfx}_map.mp4"), "rb")
            video_bytes = video_file.read()
            st.video(video_bytes)
        if anomaly_detection_bool:
            img = Image.open(Path(case.folderpath, "plots", f"{case.fpfx}_pop_an.png"))
            outputs.image(img)
            video_file = open(Path(case.folderpath, f"{case.fpfx}_andes.mp4"), "rb")
            video_bytes = video_file.read()
            st.video(video_bytes)


if __name__ == "__main__":
    main()
    print("Done!")
