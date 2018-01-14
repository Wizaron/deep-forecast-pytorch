import numpy as np
import datetime, argparse
from sklearn.preprocessing import Imputer

parser = argparse.ArgumentParser()
parser.add_argument('--input', help='input file')
parser.add_argument('--processed', help='processed file')
parser.add_argument('--output', help='output file')
args = parser.parse_args()


data = np.loadtxt(args.processed, dtype='str', delimiter=',')
org_data = np.loadtxt(args.input, dtype='str', delimiter=',')

header = data[0]
data = data[1:]
org_data = org_data[1:]

def convert2datetime(txt):
    return datetime.datetime.strptime(txt, '%Y-%m-%d %H:%M')

org_data_dates = np.array(map(convert2datetime, org_data[:, 1]))

n_hours = len(data)

for i in xrange(data.shape[1] - 1):
    station_name = header[i + 1] # Station Names
    
    print '#### ', i, station_name
    
    wind_speeds = data[:, i + 1] # Wind Speeds for that station

    empties = np.where(wind_speeds == 'None')[0] 
    ms = np.where(wind_speeds == 'M')[0]
    missing_indexes = list(empties) + list(ms)
    missing_indexes = list(set(missing_indexes)) # Missing value indexes
    
    missing_ratio = float(len(missing_indexes)) / n_hours
    
    print missing_ratio

    if missing_ratio > 0.02:
        continue

    for idx in missing_indexes:
        date = datetime.datetime.strptime(data[idx, 0], '%Y-%m-%d %H:%M:%S')
        
        org_station_indexes = org_data[:, 0] == station_name
        org_station_data = org_data[org_station_indexes]
        org_station_dates = org_data_dates[org_station_indexes]
        
        found = False
        for cntr in range(20, 30 + 1, 10): #5, 30 + 1, 5
            below_range = date - datetime.timedelta(minutes=cntr)
            upper_range = date + datetime.timedelta(minutes=cntr)
        
            for j, d in enumerate(org_station_dates):
                if (below_range < d < upper_range) and (d != date) and (org_station_data[j][-1] != 'M'):
                    replacement_date = d
                    found = True
                    break
                    
            if found:
                break
                
        if not found:
            continue
        
        data[idx, i + 1] = org_station_data[org_station_dates == replacement_date][0][-1]



new_header = []
new_data = []

n_hours = len(data)
for i in xrange(data.shape[1] - 1):
    empties = np.where(data[:, i + 1] == 'None')[0]
    ms = np.where(data[:, i + 1] == 'M')[0]
    missing_indexes = list(empties) + list(ms)
    missing_indexes = list(set(missing_indexes))
    missing_ratio = float(len(missing_indexes)) / n_hours
    if missing_ratio <= 0.01:
        print i, missing_ratio
        new_header.append(header[i + 1])
        new_data.append(data[:, i + 1])

new_data = np.stack(new_data, axis=1)
#new_data = np.concatenate((np.expand_dims(new_header, axis=0), new_data), axis=0)

new_data[new_data == 'None'] = '-1'
new_data[new_data == 'M'] = '-1'

new_data = new_data.astype('float')
imp = Imputer(missing_values=-1.0, axis=0)
new_data = imp.fit_transform(new_data)

np.savetxt(args.output, new_data, fmt='%s', delimiter=',')
