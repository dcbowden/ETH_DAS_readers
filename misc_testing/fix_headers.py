from datetime import datetime, timedelta
import glob
import h5py

start_dir = "./"
files = glob.glob(start_dir+"*h5")

sdf = '%Y-%m-%dT%H:%M:%S.%f+00:00'         # ascii format to always use
sdt = h5py.string_dtype('utf-8', 32)       # specify StringDataType

for file in files:
    print(file)
    file_start = datetime.strptime(file,'../istanbul_setup__UTC_%Y%m%d_%H%M%S.%f.h5')
    print(file_start)

    with h5py.File(file, "r+") as f:
        #subgroup_raw.attrs.create("RawDataUnit", data=headers['unit'].encode('ascii'), dtype=sdt)
        print(f['Acquisition/Raw[0]/RawData/'].attrs['PartStartTime'])# = file_start.strftime(sdf).encode('ascii')

        del f['Acquisition/Raw[0]/RawData/'].attrs['PartStartTime']
        f['Acquisition/Raw[0]/RawData/'].attrs.create("PartStartTime", data=file_start.strftime(sdf).encode('ascii'), dtype=sdt)

        del f['Acquisition/Raw[0]/RawDataTime/'].attrs['PartStartTime']
        f['Acquisition/Raw[0]/RawDataTime/'].attrs.create("PartStartTime", data=file_start.strftime(sdf).encode('ascii'), dtype=sdt)

        del f['Acquisition/Raw[0]/RawDataTime/'].attrs['StartTime']
        f['Acquisition/Raw[0]/RawDataTime/'].attrs.create("StartTime", data=file_start.strftime(sdf).encode('ascii'), dtype=sdt)

        del f['Acquisition'].attrs['MeasurementStartTime']
        f['Acquisition'].attrs.create("MeasurementStartTime", data=file_start.strftime(sdf).encode('ascii'), dtype=sdt)

        dt = 1/f['Acquisition/Raw[0]'].attrs['OutputDataRate']
        file_end = file_start+timedelta(seconds=30-dt)
        del f['Acquisition/Raw[0]/RawData/'].attrs['PartEndTime']
        f['Acquisition/Raw[0]/RawData/'].attrs.create("PartEndTime", data=file_end.strftime(sdf).encode('ascii'), dtype=sdt)

        del f['Acquisition/Raw[0]/RawDataTime/'].attrs['PartEndTime']
        f['Acquisition/Raw[0]/RawDataTime/'].attrs.create("PartEndTime", data=file_end.strftime(sdf).encode('ascii'), dtype=sdt)

        print(" --->  "+f['Acquisition/Raw[0]/RawData/'].attrs['PartStartTime'])# = file_start.strftime(sdf).encode('ascii')
