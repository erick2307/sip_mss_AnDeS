from datetime import datetime, timedelta

import os
import pandas as pd
import numpy as np
import time

Pyear = 2025
Pmonth = 1
Pday = 26 # available data until 23h

def save_data():
    t0 = time.time()
    data = {}
    from_date = datetime(year=Pyear, month=1, day=1)
    if Pmonth == 12 and Pday == 31:
        to_date = datetime(year=Pyear+1, month=1, day=1)
    else:
        to_date = datetime(year=Pyear, month=Pmonth, day=Pday+1)
    num_rows = (to_date - from_date) // timedelta(hours=1)
    cur = from_date
    i = 0
    while cur < to_date:
        path = cur.strftime("/Volumes/Pegasus32/data/NTT_Data/%Y_csv/%Y%m%d/clipped_mesh_pop_%Y%m%d%H00_00000.csv")
        print(cur)
        
        if not os.path.isfile(path):
            print('passed')
            cur += timedelta(hours=1)
            i += 1
            continue

        df = pd.read_csv(path)
        for row in df.itertuples(index=False):
            if row.area not in data:
                data[row.area] = np.full(num_rows, -1, dtype=np.int32)
            data[row.area][i] = row.population

        print(path, len(data))

        cur += timedelta(hours=1)
        i += 1

    out = np.lib.format.open_memmap(f"./data/ntt_mss_{Pyear}.npy", mode="w+", dtype=np.int32,
                                    shape=(num_rows, len(data)), fortran_order=True)

    for i, v in enumerate(data.values()):
        out[:, i] = v

    out.flush()

    np.save(f"./data/ntt_mss_{Pyear}_areas.npy", np.fromiter(data.keys(), dtype=np.int32))

    print(f'{time.time() - t0}')
    return


if __name__ == "__main__":
    save_data()