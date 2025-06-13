
import numpy as np
import stumpy
from datetime import datetime, timedelta
from icecream import ic
from tqdm import tqdm
import arrow
from dateutil import tz
import matplotlib.pyplot as plt
import joblib

timezone = tz.gettz("Asia/Tokyo")

def read_database(start=2016, stop=2024):
    db = {}
    for year in tqdm(np.arange(start, stop + 1, 1), desc="Reading database"):
        y = np.load(f"data/ntt_mss_{year}.npy")
        mids = np.load(f"data/ntt_mss_{year}_areas.npy")
        db[year] = (y, mids)
    return db

def join_database_of_meshcode(db, meshcode):
    for key in db.keys():
        meshids = db[key][1]
        try:
            pos = np.where(meshids == meshcode)[0].item()
        except:
            print(f"Meshcode {meshcode} not found in {key}")
            return None
        if key == 2016:
            data = db[key][0][:,pos]
        else:
            data = np.concatenate((data, db[key][0][:,pos]), axis=0)
    return data

def set_vars(cur):
    target_year = cur.year
    target_month = cur.month
    target_day = cur.day
    target_hour = cur.hour
    return target_year, target_month, target_day, target_hour

def get_date_one_month_ago(target_year, target_month, target_day, target_hour):
    to_date = arrow.get(target_year, target_month, target_day, target_hour).replace(tzinfo=timezone)
    last_month = to_date.shift(months=-1)
    return last_month.datetime

def end_of_year_bool(year1,year2):
    if year1 != year2:
        return True
    else:
        return False

def get_hours(start_year, start_month, start_day, start_hour, stop_year, stop_month, stop_day, stop_hour):
    start = datetime(start_year, start_month, start_day, start_hour, 0, 0, 0, tzinfo=timezone)
    stop = datetime(stop_year, stop_month, stop_day, stop_hour, 0, 0, 0, tzinfo=timezone)
    delta = stop - start
    return delta.days * 24 + delta.seconds // 3600

def get_dateindex(dt, num_days):
    index = sum(num_days[:dt.year-2016]) + get_hours(dt.year,1,1,0,dt.year,dt.month,dt.day,dt.hour)
    return index

def get_date_from_index(index, num_days):
    year = 2016
    while index >= num_days[year - 2016]:
        index -= num_days[year - 2016]
        year += 1
    month = 1
    while index >= get_hours(year, month, 1, 0, year, month + 1, 1, 0):
        index -= get_hours(year, month, 1, 0, year, month + 1, 1, 0)
        month += 1
    day = index // 24 + 1
    hour = index % 24
    return datetime(year, month, day, hour, tzinfo=timezone)

def get_history_data(data, num_days, cur, last_month, eoy_bool, alpha=0):
    if eoy_bool:
        alpha = 1
    for i, year in enumerate(np.arange(2016, last_month.year+1)):
        if cur.month == 2 and cur.day == 29 and year % 4 != 0: #<== added to handle Feb 29
            continue            
        start_index = get_dateindex(datetime(year,last_month.month,last_month.day,last_month.hour,0,0,0,timezone), num_days)
        stop_index = get_dateindex(datetime(year+alpha,cur.month,cur.day,cur.hour,0,0,0,timezone), num_days)
        if i == 0:    
            history_data = data[start_index:stop_index]
        else:
            history_data = np.concatenate((history_data, data[start_index:stop_index]), axis=0)
    return history_data[:-1]

def andes(data, num_days, cur, stream_point, meshcode):    
    target_year, target_month, target_day, target_hour = set_vars(cur)
    last_month = get_date_one_month_ago(target_year, target_month, target_day, target_hour)
    eoy_bool = end_of_year_bool(target_year, last_month.year)
    history_data = get_history_data(data, num_days, cur, last_month, eoy_bool)
    # convert data to float64 and replace -1 with np.nan
    history_data = np.where(history_data == -1, np.nan, history_data).astype(np.float64)
    stream_point = np.where(stream_point == -1, np.nan, stream_point).astype(np.float64)
    # calculate the left matrix profile
    T = history_data
    m = 3
    stream = stumpy.stumpi(T, m, normalize=False, egress=False)
    # update stream with new data
    stream.update(stream_point)
    # calculate the left MP
    a = stream._left_P[m:]
    # filter initial 'inf' values
    af = a[np.isfinite(a)]
    # calculate std of the left MP ==> Condition for anomaly detection
    factor = 3
    # print(f"Factor: {factor}")
    # print(f"Max: {af.max()}, Min: {af.min()}")
    # print(f"Mean: {af.mean()}, Std: {af.std()}")
    return T, stream._left_P[-1] > (af.mean() + int(factor) * af.std())


# YOU MAY RUN THIS ONCE TO LOAD THE DATABASE
db = read_database(start=2016, stop=2024)

#===== ONE EVENT MANY MESHESCODES ==========================================================
event_dt = datetime(2022,3,16,23,0,0,0,timezone) # Fukushima EQ
meshcodes = [ 574007614,  574007623,  574007624,  574007633,  574007634,
574007612,  574007621,  574007622,  574007631,  574007632, 
574007514,  574007523,  574007524,  574007533,  574007534,
574007512,  574007521,  574007522,  574007531,  574007532,
574007414,  574007423,  574007424,  574007433,  574007434,
574007412,  574007421,  574007422,  574007431,  574007432,
574007314,  574007323,  574007324,  574007333,  574007334 ]
event = 'Fukushima EQ'
#====== END OF EVENT ==========================================================

for meshcode in meshcodes:
    ic(meshcode)
    # ALWAYS NEEDED ===================================================================
    title = f"Meshcode: {meshcode}, Date: {event_dt.strftime('%Y-%m-%d %H:%M')}, {event}"
    filename = f"andes_{event.replace(' ','')}_{meshcode}_{event_dt.strftime('%Y%m%d%H%M')}"
    data = join_database_of_meshcode(db, meshcode)
    # create a list with shape of database for each year, number of hours per year
    num_days = [db[key][0].shape[0] for key in db.keys()]

    xhours = 14 * 24
    dates_before = [event_dt - timedelta(hours=i) for i in range(xhours,0,-1)]
    dates_after = [event_dt + timedelta(hours=i) for i in range(xhours)]
    # dates = dates_before + dates_after
    dates = dates_after
    img = []
    for date in dates:
        # this is input data
        cur_index = get_dateindex(date, num_days)
        stream_point = data[cur_index]
        # online anomaly detection
        T, anomaly = andes(data, num_days, date, stream_point, meshcode)
        img.append(anomaly)
        # if anomaly:
        #     ic(date)

    arr = np.array(img).astype(int)
    # add to arr cur.hours points of np.nan at the beginning
    start_pad = np.full(dates[0].hour, np.nan)
    arr = np.concatenate((start_pad, arr))
    end_pad = np.full((23 - dates[-1].hour), np.nan)
    arr = np.concatenate((arr,end_pad))
    # add the current point to the array
    # arr[len(start_pad)+len(dates_before)] = 2
    arr[len(start_pad)] = 2
    
    plt.figure(figsize=(10, 5))
    plt.imshow(arr.reshape(-1,24), cmap='CMRmap_r', aspect=0.7)
    days = set([dates[i].date() for i in range(len(dates))])
    # set up y ticks labels
    plt.yticks(np.arange(0,len(days)), [f"{i}" for i in sorted(days)])
    plt.xticks(np.arange(0,24), [f"{i:02d}" for i in np.arange(0,24)], rotation=45);
    # create vertical lines at 0.5 of each hour
    for i in range(1,24):
        plt.axvline(i-0.5, color='black', linewidth=0.3)
    # create horizontal lines at 0.5 of each hour
    for i in range(1,len(days)):
        plt.hlines(i-0.5, xmin=-0.5,xmax=23.5,color='black', linewidth=0.3)
    plt.xlabel("Hour Frame of the day (e.g. 08 means 08:00 to 08:59)")
    plt.title(title)
    plt.savefig(f"{filename}.png", dpi=300)
    plt.close()