import numpy as np
import datetime, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', help='input file')
parser.add_argument('--output', help='output file')
args = parser.parse_args()


data = np.loadtxt(args.input, dtype='str', delimiter=',')
header = data[0]
data = data[1:]

def convert2datetime(txt):
    return datetime.datetime.strptime(txt, '%Y-%m-%d %H:%M')

dates = np.array(map(convert2datetime, data[:,1]))
data = data[:, [0, 2]]

#discarded_stations = ['XNT', '6B9', 'OLE', 'MTP', 'N03', 'NY0', 'OGS']
discarded_stations = []

stations = list(set(data[:, 0]).difference(discarded_stations))

all_dates = set()
all_data = []
for station in stations:
    station_filter = data[:, 0] == station

    station_winds = data[station_filter][:, 1]
    station_dates = dates[station_filter]

    sort_idxes = station_dates.argsort()
    station_winds = station_winds[sort_idxes]
    station_dates = station_dates[sort_idxes]

    select_idxes = np.array([i for i in xrange(len(station_dates)) if station_dates[i].minute == 50]) #TODO: 0

    if len(select_idxes) == 0:
        print station
        continue

    station_winds = station_winds[select_idxes]
    station_dates = station_dates[select_idxes]

    #print '##############'
    #print station
    #print station_winds, station_dates

    all_data.append([station, station_dates, station_winds])

    for s in station_dates:
        all_dates.add(s)

all_dates = list(all_dates)
all_dates.sort()

last_date = max(all_dates)
first_date = min(all_dates)
diff = last_date - first_date

n_hours = int(diff.total_seconds() / (60. * 60.))
date_axis = np.array([first_date + datetime.timedelta(hours=i) for i in range(n_hours + 1)])

data_axis = []
for data in all_data:
    station = data[0]
    dates = np.array(data[1])
    winds = np.array(data[2])

    winds_axis = []
    for date in date_axis:
        idxes = np.where(dates == date)[0]
        assert (len(idxes) in [0, 1])
        if len(idxes) == 0:
            winds_axis.append(None)
        else:
            idx = idxes[0]
            winds_axis.append(winds[idx])
    data_axis.append([station, winds_axis])

new_data = []
for station, winds in data_axis[1:]:
    _data = [station, ] + winds
    new_data.append(_data)

out = np.concatenate((np.expand_dims(['date', ] + list(date_axis.astype('str')), axis=1), np.stack(new_data, axis=1)), axis=1)

np.savetxt(args.output, out, fmt='%s', delimiter=',')
